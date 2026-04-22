import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

/** Standard shadcn `cn()` helper: merge Tailwind classes safely. */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
