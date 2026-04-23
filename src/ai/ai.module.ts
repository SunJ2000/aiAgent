import { Module } from '@nestjs/common';
import { AiService } from './ai.service';
import { AiController } from './ai.controller';
import { ConfigService } from '@nestjs/config';
import { ChatOpenAI } from '@langchain/openai';
import z from 'zod';
import { tool } from '@langchain/core/tools';
import { MailerService } from '@nestjs-modules/mailer';


@Module({
  controllers: [AiController],
  providers: [
    AiService,
    {
      provide: 'CHAT_MODEL',
      useFactory: (configService: ConfigService) => {
        return new ChatOpenAI({
          temperature: 0.7,
          modelName: configService.get('MODEL_NAME'),
          apiKey: configService.get('OPENAI_API_KEY'),
          configuration: {
            baseURL: configService.get('OPENAI_BASE_URL'),
          },
        });
      },
      inject: [ConfigService],
    },
    {
      provide: "SEND_EMAIL_TOOL",
      useFactory: (mailerService: MailerService, configService: ConfigService) => {
        const sendEmailArgsSchema = z.object({
          to: z.string().email().describe('收件人邮箱'),
          subject: z.string().describe('邮件主题'),
          text: z.string().optional().describe('纯文本内容，可选'),
          html: z.string().optional().describe('HTML 内容，可选'),
        });

        return tool(async ({ to, subject, text, html }: { to: string, subject: string, text?: string, html?: string }) => {
          const fallbackFrom = configService.get('MAIL_FROM');
          await mailerService.sendMail({
            to,
            subject,
            text: text ?? '（无文本内容）',
            html: html ?? `<p>${text ?? '（无 HTML 内容）'}</p>`,
            from: fallbackFrom,
          });

          return `已发送邮件至 ${to}，主题为 ${subject}`;
        },
          {
            name: 'send_email',
            description:
              '发送邮件。输入收件人邮箱、邮件主题、纯文本内容（可选）和 HTML 内容（可选），返回发送结果。',
            schema: sendEmailArgsSchema,

          })

      },
      inject: [MailerService, ConfigService],
    },
    {
      provide: "WEB_SEARCH_TOOL",
      useFactory: (configService: ConfigService) => {
        const webSearchArgsSchema = z.object({
          query: z.string().describe('搜索关键词'),
          count: z.number().int().min(1).max(20).optional().describe('返回的搜索结果数量，默认10条'),
        });

        return tool(async ({ query, count = 10 }: { query: string, count?: number }) => {

          const apiKey = configService.get<string>('BOCHA_API_KEY');

          if (!apiKey) {
            return 'Bocha Web Search 的 API Key 未配置（环境变量 BOCHA_API_KEY），请先在服务端配置后再重试。';
          }
          const url = 'https://api.bochaai.com/v1/web-search';
          const body = {
            query,
            freshness: 'noLimit',
            summary: true,
            count,
          }
          const response = await fetch(url, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${apiKey}`,
            },
            body: JSON.stringify(body),
          });

          if (!response.ok) {
            const errorText = await response.text();
            return `搜索失败，请稍后再试。错误信息：${errorText}`;
          }

          let json: any;
          try {
            json = await response.json();

          } catch (error) {
            return `搜索失败，请稍后再试。错误信息：${error}`;
          }

          try {
            if (json.code !== 200 || !json.data) {
              return `搜索 API 请求失败，原因是: ${json.msg ?? '未知错误'}`;
            }
            const webpages = json.data.webPages?.value ?? [];
            if (!webpages.length) {
              return '没有找到相关结果。';
            }

            const formatted = webpages.map((page: any, idx: number) => `引用: ${idx + 1}
标题: ${page.name}
URL: ${page.url}
摘要: ${page.summary}
网站名称: ${page.siteName}
网站图标: ${page.siteIcon}
发布时间: ${page.dateLastCrawled}`).join('\n\n');

            return formatted;
          } catch (error) {
            return `搜索失败，请稍后再试。错误信息：${error}`;
          }

        },
          {
            name: 'web_search',
            description:
              '使用 Bocha Web Search 搜索。输入搜索关键词，返回搜索结果。',
            schema: webSearchArgsSchema,
          })
      },
      inject: [ConfigService],
    },
  ],
})
export class AiModule { }
