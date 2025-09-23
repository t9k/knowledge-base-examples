import copy
import json
import re
from typing import Dict, Iterator, List, Literal, Optional, Union

from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm.schema import FUNCTION, Message


class LawQaAgent(FnCallAgent):
    """继承自 FnCallAgent 的法律问答智能体，添加了重排工具的特殊处理逻辑"""

    def _run(self, messages: List[Message], lang: Literal['en', 'zh'] = 'en', **kwargs) -> Iterator[List[Message]]:
        """重写 _run 方法以支持重排工具的特殊处理"""
        messages = copy.deepcopy(messages)
        from qwen_agent.settings import MAX_LLM_CALL_PER_RUN
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        response = []
        # Global counter for <source id="n"> blocks across multiple tool calls in one run
        global_source_id_counter: int = 0
        
        while True and num_llm_calls_available > 0:
            num_llm_calls_available -= 1

            extra_generate_cfg = {'lang': lang}
            if kwargs.get('seed') is not None:
                extra_generate_cfg['seed'] = kwargs['seed']
            output_stream = self._call_llm(messages=messages,
                                           functions=[func.function for func in self.function_map.values()],
                                           extra_generate_cfg=extra_generate_cfg)
            output: List[Message] = []
            for output in output_stream:
                if output:
                    yield response + output
            if output:
                response.extend(output)
                messages.extend(output)
                used_any_tool = False
                
                # Check if there's a reranker-rerank tool call in output
                has_rerank_tool = False
                rerank_tool_calls = []
                other_tool_calls = []
                
                for out in output:
                    use_tool, tool_name, tool_args, _ = self._detect_tool(out)
                    if use_tool:
                        if tool_name == 'reranker-rerank':
                            has_rerank_tool = True
                            rerank_tool_calls.append((out, tool_name, tool_args))
                        else:
                            other_tool_calls.append((out, tool_name, tool_args))
                
                if has_rerank_tool:
                    # Special handling for reranker-rerank: collect search results in buffer
                    search_results_buffer = []
                    
                    # First execute all non-rerank tools and collect their results
                    for out, tool_name, tool_args in other_tool_calls:
                        tool_result = self._call_tool(tool_name, tool_args, messages=messages, **kwargs)
                        # Renumber <source id="..."> blocks to be globally unique within this run
                        try:
                            tool_result, inc = self._renumber_source_blocks(tool_result, start_index=global_source_id_counter + 1)
                            global_source_id_counter += inc
                        except Exception:
                            # If anything goes wrong during renumbering, fall back to original content
                            pass
                        
                        # Add to buffer for later use in rerank
                        search_results_buffer.append(tool_result)
                        
                        # Create Message with placeholder content and add to messages/response
                        fn_msg = Message(role=FUNCTION,
                                         name=tool_name,
                                         content="见重排结果",
                                         extra={'function_id': out.extra.get('function_id', '1')})
                        
                        messages.append(fn_msg)
                        response.append(fn_msg)
                        yield response
                        used_any_tool = True
                    
                    # Then execute rerank tools with concatenated search results
                    for out, tool_name, tool_args in rerank_tool_calls:
                        # Parse tool_args and add concatenated search results
                        if isinstance(tool_args, str):
                            args_dict = json.loads(tool_args)
                        else:
                            args_dict = tool_args.copy()
                        
                        # Concatenate all search results as search_results parameter
                        concatenated_results = '\n\n'.join(search_results_buffer)
                        args_dict['search_results'] = concatenated_results
                        
                        # Call rerank tool with modified arguments
                        tool_result = self._call_tool(tool_name, json.dumps(args_dict), messages=messages, **kwargs)
                        
                        # Create Message for rerank result and add to messages/response
                        fn_msg = Message(role=FUNCTION,
                                         name=tool_name,
                                         content=tool_result,
                                         extra={'function_id': out.extra.get('function_id', '1')})
                        
                        messages.append(fn_msg)
                        response.append(fn_msg)
                        yield response
                        used_any_tool = True
                else:
                    # Original logic for cases without reranker-rerank
                    for out in output:
                        use_tool, tool_name, tool_args, _ = self._detect_tool(out)
                        if use_tool:
                            tool_result = self._call_tool(tool_name, tool_args, messages=messages, **kwargs)
                            # Renumber <source id="..."> blocks to be globally unique within this run
                            try:
                                tool_result, inc = self._renumber_source_blocks(tool_result, start_index=global_source_id_counter + 1)
                                global_source_id_counter += inc
                            except Exception:
                                # If anything goes wrong during renumbering, fall back to original content
                                pass
                            fn_msg = Message(role=FUNCTION,
                                             name=tool_name,
                                             content=tool_result,
                                             extra={'function_id': out.extra.get('function_id', '1')})
                            
                            messages.append(fn_msg)
                            response.append(fn_msg)
                            yield response
                            used_any_tool = True
                if not used_any_tool:
                    break
        yield response

    def _renumber_source_blocks(self, text: str, start_index: int) -> tuple[str, int]:
        """Renumber <source id="n"> blocks so that ids are globally unique.

        Args:
            text: The tool result text that may contain multiple <source id="..."> blocks.
            start_index: The starting index (1-based) to assign to the first encountered source block.

        Returns:
            A pair of (new_text, count) where new_text has ids rewritten to be
            consecutive starting from start_index, and count is the number of
            source blocks found and rewritten.
        """
        if not isinstance(text, str) or '<source' not in text:
            return text, 0

        pattern = re.compile(r"(<source\s+id=\")([0-9]+)(\">)")

        # We will replace sequentially in appearance order with start_index..
        current = start_index
        count = 0

        def _repl(m: 're.Match[str]') -> str:
            nonlocal current, count
            count += 1
            repl = f"{m.group(1)}{current}{m.group(3)}"
            current += 1
            return repl

        new_text = pattern.sub(_repl, text)
        return new_text, count
