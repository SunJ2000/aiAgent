import { Inject, Injectable } from '@nestjs/common';
import { BaseMessage, getBufferString, SystemMessage } from '@langchain/core/messages';
import { ChatOpenAI } from '@langchain/openai';

const MAX_MESSAGES = 10;
const KEEP_RECENT = 4;

interface Session {
  messages: BaseMessage[];
  memory: string;
}

@Injectable()
export class MemoryService {
  private sessions = new Map<string, Session>();

  constructor(@Inject('CHAT_MODEL') private readonly model: ChatOpenAI) {}

  getSession(sessionId: string): Session {
    if (!this.sessions.has(sessionId)) {
      this.sessions.set(sessionId, { messages: [], memory: '' });
    }
    return this.sessions.get(sessionId)!;
  }

  getMemoryPrompt(sessionId: string): SystemMessage | null {
    const session = this.getSession(sessionId);
    if (!session.memory) return null;
    return new SystemMessage(`以下是之前对话的记忆摘要，请参考：\n${session.memory}`);
  }

  pushMessages(sessionId: string, ...msgs: BaseMessage[]) {
    const session = this.getSession(sessionId);
    session.messages.push(...msgs);
  }

  async trimIfNeeded(sessionId: string): Promise<void> {
    const session = this.getSession(sessionId);
    if (session.messages.length <= MAX_MESSAGES) return;

    const toSummarize = session.messages.slice(0, -KEEP_RECENT);
    session.memory = await this.summarize(toSummarize, session.memory);
    session.messages.splice(0, session.messages.length - KEEP_RECENT);
  }

  private async summarize(messages: BaseMessage[], existingMemory: string): Promise<string> {
    const conversationText = getBufferString(messages, '用户', '助手');
    const prompt = existingMemory
      ? `以下是之前的记忆摘要：\n${existingMemory}\n\n以下是新的对话内容：\n${conversationText}\n\n请将之前的记忆和新对话合并总结，保留所有重要信息：`
      : `请总结以下对话的核心内容，保留重要信息：\n\n${conversationText}\n\n总结：`;

    const response = await this.model.invoke([new SystemMessage(prompt)]);
    return response.content as string;
  }

  clearSession(sessionId: string) {
    this.sessions.delete(sessionId);
  }
}
