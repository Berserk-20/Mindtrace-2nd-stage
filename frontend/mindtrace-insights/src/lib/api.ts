
// Use environment variable for production, fallback to localhost for dev
export const API_BASE_URL = "https://gaikwadsankalp-1424-mindtrace-backend.hf.space";

/**
 * Get authorization headers with JWT token
 */
const getAuthHeaders = () => {
  const token = localStorage.getItem("token");
  return {
    "Authorization": `Bearer ${token}`,
    "Content-Type": "application/json",
    "ngrok-skip-browser-warning": "true"
  };
};

export const fetchMetrics = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/metrics`, {
      headers: getAuthHeaders()
    });

    if (response.status === 401) {
      // Token expired or invalid - redirect to login
      window.location.href = "/login";
      return null;
    }

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Failed to fetch metrics:", error);
    return null;
  }
};

export const fetchLiveMetrics = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/metrics/live`, {
      headers: getAuthHeaders()
    });

    if (response.status === 401) {
      window.location.href = "/login";
      return null;
    }

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Failed to fetch live metrics:", error);
    return null;
  }
};

export const startSession = async () => {
  try {
    console.log("Starting session using API:", API_BASE_URL);
    const res = await fetch(`${API_BASE_URL}/start`, {
      headers: getAuthHeaders()
    });
    if (res.status === 401) {
      window.location.href = "/login";
      return null;
    }
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Failed to start session");
    }
    return res.json();
  } catch (error) {
    console.error("Error starting session:", error);
    throw error;
  }
};

export const pauseSession = async () => {
  try {
    const res = await fetch(`${API_BASE_URL}/pause`, {
      headers: getAuthHeaders()
    });
    if (res.status === 401) {
      window.location.href = "/login";
      return null;
    }
    return res.json();
  } catch (error) {
    console.error("Error pausing session:", error);
    throw error;
  }
};

export const resumeSession = async () => {
  try {
    const res = await fetch(`${API_BASE_URL}/resume`, {
      headers: getAuthHeaders()
    });
    if (res.status === 401) {
      window.location.href = "/login";
      return null;
    }
    return res.json();
  } catch (error) {
    console.error("Error resuming session:", error);
    throw error;
  }
};

export const stopSession = async () => {
  try {
    const res = await fetch(`${API_BASE_URL}/stop`, {
      headers: getAuthHeaders()
    });
    if (res.status === 401) {
      window.location.href = "/login";
      return null;
    }
    return res.json();
  } catch (error) {
    console.error("Error stopping session:", error);
    throw error;
  }
};

export const fetchSessions = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/sessions`, {
      headers: getAuthHeaders()
    });

    if (response.status === 401) {
      window.location.href = "/login";
      return null;
    }

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Failed to fetch sessions:", error);
    return null;
  }
};

export const fetchEmotions = async (sessionId: string) => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/emotions?session_id=${sessionId}`, {
      headers: getAuthHeaders()
    });

    if (response.status === 401) {
      window.location.href = "/login";
      return null;
    }

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Failed to fetch emotions:", error);
    return null;
  }
};

export const fetchUserProfile = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/me`, {
      headers: getAuthHeaders()
    });

    if (response.status === 401) {
      window.location.href = "/login";
      return null;
    }

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Failed to fetch user profile:", error);
    return null;
  }
};

export const fetchReports = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/reports`, {
      headers: getAuthHeaders()
    });

    if (response.status === 401) {
      window.location.href = "/login";
      return null;
    }

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Failed to fetch reports:", error);
    return null;
  }
};

export const fetchAlerts = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/alerts`, {
      headers: getAuthHeaders()
    });

    if (response.status === 401) {
      window.location.href = "/login";
      return null;
    }

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Failed to fetch alerts:", error);
    return null;
  }
};
