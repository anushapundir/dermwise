import Hero from "@/components/Hero";
import Workflow from "@/components/Workflow";
import Disclaimer from "@/components/Disclaimer";

/** Landing page — combines Hero, Workflow steps, and Disclaimer. */
export default function HomePage() {
  return (
    <>
      <Hero />
      <Workflow />
      <Disclaimer />
    </>
  );
}
