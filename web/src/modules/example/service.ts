import { api } from "@/lib/api";
import type { Example } from "./types";

export const listExamples = () => api<Example[]>("/examples");
