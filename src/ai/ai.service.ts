import { AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage } from '@langchain/core/messages';
import { Runnable } from '@langchain/core/runnables';
import { ChatOpenAI } from '@langchain/openai';
import { Inject, Injectable } from '@nestjs/common';
import { MemoryService } from './memory.service';

@Injectable()
export class AiService {
  private readonly modelWithTools: Runnable<BaseMessage[], AIMessage>;
  private readonly tools: any[];

  constructor(@Inject('CHAT_MODEL') model: ChatOpenAI,
    @Inject('SEND_EMAIL_TOOL') private readonly sendMailTool: any,
    @Inject('WEB_SEARCH_TOOL') private readonly webSearchTool: any,
    @Inject('RAG_SEARCH_TOOL') private readonly ragSearchTool: any,
    private readonly memoryService: MemoryService) {
    this.tools = [this.sendMailTool, this.webSearchTool, this.ragSearchTool];
    this.modelWithTools = model.bindTools(this.tools);
  }

  async runChain(query: string, sessionId = 'default'): Promise<string> {
    const messages: BaseMessage[] = [
      new SystemMessage(
        '你是一个智能助手，可以在需要时调用工具，再用结果回答用户的问题。',
      ),
    ];

    const memoryPrompt = this.memoryService.getMemoryPrompt(sessionId);
    if (memoryPrompt) {
      messages.push(memoryPrompt);
    }

    const session = this.memoryService.getSession(sessionId);
    messages.push(...session.messages);

    const humanMsg = new HumanMessage(query);
    messages.push(humanMsg);
    this.memoryService.pushMessages(sessionId, humanMsg);

    while (true) {
      const aiMessage = await this.modelWithTools.invoke(messages);
      messages.push(aiMessage);
      const toolCalls = aiMessage.tool_calls ?? [];
      if (!toolCalls.length) {
        this.memoryService.pushMessages(sessionId, aiMessage);
        await this.memoryService.trimIfNeeded(sessionId);
        return aiMessage.content as string;
      }

      const toolResults = await Promise.all(
        toolCalls.map(async (toolCall) => {
          const tool = this.tools.find((t) => t.name === toolCall.name);
          if (!tool) {
            throw new Error(`No tool found with name ${toolCall.name}`);
          }
          try {
            return await tool.invoke(toolCall.args);
          } catch (e: any) {
            return `error: ${e.message}`;
          }
        }),
      );

      toolCalls.forEach((toolCall, index) => {
        messages.push(
          new ToolMessage({
            tool_call_id: toolCall.id || '',
            name: toolCall.name,
            content: toolResults[index],
          }),
        );
      });
    }
  }

  async *streamChain(query: string, sessionId = 'default'): AsyncGenerator<string> {
    const messages: BaseMessage[] = [
      new SystemMessage(
        '你是一个智能助手，可以在需要时调用工具，再用结果回答用户的问题。',
      ),
    ];

    const memoryPrompt = this.memoryService.getMemoryPrompt(sessionId);
    if (memoryPrompt) {
      messages.push(memoryPrompt);
    }

    const session = this.memoryService.getSession(sessionId);
    messages.push(...session.messages);

    const humanMsg = new HumanMessage(query);
    messages.push(humanMsg);
    this.memoryService.pushMessages(sessionId, humanMsg);

    while (true) {
      const stream = await this.modelWithTools.stream(messages);
      let fullAiMessage: AIMessageChunk | null = null;
      for await (const chunk of stream as AsyncIterable<AIMessageChunk>) {
        fullAiMessage = fullAiMessage ? fullAiMessage.concat(chunk) : chunk;
        const hasTollCallChunk = !!fullAiMessage.tool_call_chunks && fullAiMessage.tool_call_chunks.length > 0;
        if (!hasTollCallChunk && chunk.content) {
          yield chunk.content as string;
        }
      }
      if (!fullAiMessage) {
        return;
      }
      messages.push(fullAiMessage);

      const toolCalls = fullAiMessage.tool_calls ?? [];

      if (!toolCalls.length) {
        this.memoryService.pushMessages(sessionId, fullAiMessage);
        await this.memoryService.trimIfNeeded(sessionId);
        return;
      }

      const toolResults = await Promise.all(
        toolCalls.map(async (toolCall) => {
          const tool = this.tools.find((t) => t.name === toolCall.name);
          if (!tool) {
            throw new Error(`No tool found with name ${toolCall.name}`);
          }
          try {
            return await tool.invoke(toolCall.args);
          } catch (e: any) {
            return `error: ${e.message}`;
          }
        }),
      );

      toolCalls.forEach((toolCall, index) => {
        messages.push(
          new ToolMessage({
            tool_call_id: toolCall.id || '',
            name: toolCall.name,
            content: toolResults[index],
          }),
        );
      });

    }
  }
}
