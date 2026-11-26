import React from "react";

export interface BrandMarkProps {
  size?: number;
  className?: string;
}

export const BrandMark: React.FC<BrandMarkProps> = ({ size = 22, className }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 32 32"
    fill="none"
    aria-hidden="true"
    className={className ? `brand-mark ${className}` : "brand-mark"}
    data-testid="brand-mark-svg"
  >
    <defs>
      <linearGradient id="ddbr-brand-grad" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stopColor="#2563eb" />
        <stop offset="100%" stopColor="#0d9488" />
      </linearGradient>
    </defs>
    <rect width="32" height="32" rx="8" fill="url(#ddbr-brand-grad)" />
    <path
      d="M8.5 11.5V9.5C8.5 8.95 8.95 8.5 9.5 8.5H11.5M20.5 8.5H22.5C23.05 8.5 23.5 8.95 23.5 9.5V11.5M23.5 20.5V22.5C23.5 23.05 23.05 23.5 22.5 23.5H20.5M11.5 23.5H9.5C8.95 23.5 8.5 23.05 8.5 22.5V20.5"
      stroke="#ffffff"
      strokeWidth="2"
      strokeLinecap="round"
    />
    <rect x="11.5" y="12" width="9" height="2.4" rx="1.2" fill="#ffffff" />
    <rect x="11.5" y="16" width="5.5" height="2.4" rx="1.2" fill="#99f6e4" />
    <circle cx="19.5" cy="17.2" r="1.4" fill="#fde68a" />
    <rect x="11.5" y="20" width="7.2" height="2" rx="1" fill="#bfdbfe" />
  </svg>
);
