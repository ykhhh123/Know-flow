"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { cn } from "@/lib/utils";

interface MarkdownMessageProps {
  content: string;
  citationNumbers?: number[];
  citationBaseId?: string;
  onCitationClick?: (citationNumber: number) => void;
}

export function MarkdownMessage({
  content,
  citationNumbers = [],
  citationBaseId,
  onCitationClick,
}: MarkdownMessageProps) {
  const citationSet = new Set(citationNumbers);
  const renderedContent =
    citationBaseId && citationSet.size > 0
      ? content.replace(/\[(\d+)\]/g, (match, value) => {
          const citationNumber = Number(value);
          if (!citationSet.has(citationNumber)) return match;
          return `[${match}](#${citationBaseId}-citation-${citationNumber})`;
        })
      : content;

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        h1: ({ className, ...props }) => (
          <h1
            className={cn("mb-4 mt-6 text-2xl font-semibold first:mt-0", className)}
            {...props}
          />
        ),
        h2: ({ className, ...props }) => (
          <h2
            className={cn("mb-3 mt-5 text-xl font-semibold first:mt-0", className)}
            {...props}
          />
        ),
        h3: ({ className, ...props }) => (
          <h3
            className={cn("mb-2.5 mt-4 text-lg font-semibold first:mt-0", className)}
            {...props}
          />
        ),
        h4: ({ className, ...props }) => (
          <h4
            className={cn("mb-2 mt-4 text-base font-semibold first:mt-0", className)}
            {...props}
          />
        ),
        p: ({ className, ...props }) => (
          <p className={cn("my-3 leading-7 first:mt-0 last:mb-0", className)} {...props} />
        ),
        strong: ({ className, ...props }) => (
          <strong className={cn("font-semibold", className)} {...props} />
        ),
        ul: ({ className, ...props }) => (
          <ul className={cn("my-3 list-disc space-y-1 pl-6", className)} {...props} />
        ),
        ol: ({ className, ...props }) => (
          <ol className={cn("my-3 list-decimal space-y-1 pl-6", className)} {...props} />
        ),
        li: ({ className, ...props }) => (
          <li className={cn("leading-7", className)} {...props} />
        ),
        blockquote: ({ className, ...props }) => (
          <blockquote
            className={cn(
              "my-4 border-l-4 border-gray-300 pl-4 text-gray-600 dark:border-gray-600 dark:text-gray-300",
              className,
            )}
            {...props}
          />
        ),
        code: ({ className, ...props }) => (
          <code
            className={cn(
              "rounded bg-gray-100 px-1.5 py-0.5 font-mono text-[0.9em] dark:bg-gray-900",
              className,
            )}
            {...props}
          />
        ),
        pre: ({ className, ...props }) => (
          <pre
            className={cn(
              "my-4 overflow-x-auto rounded-lg bg-gray-950 p-4 text-sm leading-6 text-gray-100",
              className,
            )}
            {...props}
          />
        ),
        table: ({ className, ...props }) => (
          <div className="my-4 overflow-x-auto">
            <table
              className={cn(
                "w-full border-collapse text-left text-sm",
                className,
              )}
              {...props}
            />
          </div>
        ),
        th: ({ className, ...props }) => (
          <th
            className={cn(
              "border border-gray-200 bg-gray-50 px-3 py-2 font-semibold dark:border-gray-700 dark:bg-gray-900",
              className,
            )}
            {...props}
          />
        ),
        td: ({ className, ...props }) => (
          <td
            className={cn(
              "border border-gray-200 px-3 py-2 align-top dark:border-gray-700",
              className,
            )}
            {...props}
          />
        ),
        a: ({ className, href, children, ...props }) => {
          const citationMatch = href?.startsWith("#")
            ? href.match(/-citation-(\d+)$/)
            : null;

          if (citationMatch) {
            const citationNumber = Number(citationMatch[1]);
            return (
              <button
                type="button"
                className={cn(
                  "mx-0.5 inline-flex h-5 min-w-5 items-center justify-center rounded-full border border-blue-200 bg-blue-50 px-1.5 text-xs font-semibold text-blue-700 align-baseline hover:bg-blue-100 dark:border-blue-800 dark:bg-blue-950/40 dark:text-blue-300 dark:hover:bg-blue-900/50",
                  className,
                )}
                onClick={() => onCitationClick?.(citationNumber)}
              >
                {children}
              </button>
            );
          }

          return (
            <a
              className={cn("text-blue-600 underline underline-offset-4 dark:text-blue-400", className)}
              href={href}
              target="_blank"
              rel="noreferrer"
              {...props}
            >
              {children}
            </a>
          );
        },
      }}
    >
      {renderedContent}
    </ReactMarkdown>
  );
}
