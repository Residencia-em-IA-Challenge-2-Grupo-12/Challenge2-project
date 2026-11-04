import { listExamples } from "../service";

export async function ExampleList() {
  const items = await listExamples().catch(() => []);
  return (
    <ul>
      {items.map((i) => (
        <li key={i.id}>{i.name}</li>
      ))}
    </ul>
  );
}
