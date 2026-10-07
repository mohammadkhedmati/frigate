import { cn } from "@/lib/utils";

type LogoProps = {
  className?: string;
};
export default function Logo({ className }: LogoProps) {
  return (
    <svg
      viewBox="0 0 512 512"
      className={cn("stroke-current", className)}
      fill="none"
      strokeWidth={64}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M112 374V138l144 160 144-160v236" />
    </svg>
  );
}
