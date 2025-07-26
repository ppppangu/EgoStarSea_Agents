-- 创建智能体注册与事件跟踪表
CREATE TABLE IF NOT EXISTS public.agents (
    user_email VARCHAR(255) PRIMARY KEY,
    activate BOOLEAN NOT NULL DEFAULT true,
    export_agent_url TEXT[] NOT NULL DEFAULT '{}',
    events JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc'::text, NOW()) NOT NULL
);

-- 触发器：更新 updated_at 字段
CREATE OR REPLACE FUNCTION update_agents_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = TIMEZONE('utc'::text, NOW());
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_update_agents_updated_at ON public.agents;
CREATE TRIGGER trg_update_agents_updated_at
BEFORE UPDATE ON public.agents
FOR EACH ROW EXECUTE FUNCTION update_agents_updated_at();

-- 启用行级安全，用户只能操作自己的行
ALTER TABLE public.agents ENABLE ROW LEVEL SECURITY;

-- 仅允许对应邮箱的用户进行增删改查
CREATE POLICY "agent_owner_crud" ON public.agents
    USING (auth.email() = user_email)
    WITH CHECK (auth.email() = user_email);

-- 公共只读策略（仅可读取 activate 为 true 的记录，用于发现）
CREATE POLICY "public_read_active_agents" ON public.agents
    FOR SELECT USING (activate = true);

-- 索引
CREATE INDEX IF NOT EXISTS idx_agents_activate ON public.agents(activate); 