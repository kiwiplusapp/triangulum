import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Vault Console",
  description:
    "Macro agent console: signals, calibration, and the capital gate that derives position size from measured skill.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
