// src/api.js
//const API_URL = import.meta.env.VITE_API_URL || "/api";

export class ApiError extends Error {
  constructor(message, { status = 0, code = null, retryable = false } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.retryable = retryable;
  }
}

async function parseError(response) {
  let payload = null;
  try {
    payload = await response.json();
  } catch {
  }
  const detail = payload?.detail ?? payload;
  let message = null;

  if (typeof detail === "string" && detail.trim()) {
    message = detail.trim();
  } else if (detail && typeof detail === "object") {
    message = detail.message || null;
  }

  const code = detail && typeof detail === "object" ? detail.code : null;

  if (!message) {
    if (response.status === 503) message = "Сервис временно недоступен. Попробуйте ещё раз через минуту.";
    else if (response.status === 422) message = "Некорректный запрос. Проверьте формулировку.";
    else if (response.status === 404) message = "Запрошенный ресурс не найден.";
    else if (response.status >= 500) message = "Внутренняя ошибка сервиса. Попробуйте ещё раз.";
    else message = `Ошибка сервера (${response.status}).`;
  }

  const retryable =
    code === "llm_unavailable" ||
    response.status === 503 ||
    response.status === 504 ||
    response.status >= 500;

  return new ApiError(message, { status: response.status, code, retryable });
}

async function request(url, options) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (e) {
    // Сеть: бэк не запущен, DNS, CORS, обрыв
    throw new ApiError("Нет соединения с сервером.", {
      retryable: true,
    });
  }
  if (!response.ok) throw await parseError(response);
  try {
    return await response.json();
  } catch {
    throw new ApiError("Сервер вернул некорректный ответ.", { status: response.status });
  }
}

export async function sendChat(messages) {
  return request(`/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  });
}

export async function listDocuments() {
  return request(`/api/documents`, { method: "GET" });
}

export async function getDocument(docId) {
  return request(`/api/documents/${encodeURIComponent(docId)}`, { method: "GET" });
}

export async function health() {
  return request(`/api/health`, { method: "GET" });
}