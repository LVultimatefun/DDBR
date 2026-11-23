import React, { useMemo, useRef, useState } from "react";

export interface FacetComboboxProps {
  value: string;
  onChange: (next: string) => void;
  options: string[];
  placeholder: string;
  ariaLabel?: string;
  testId?: string;
}

export const FacetCombobox: React.FC<FacetComboboxProps> = ({
  value,
  onChange,
  options,
  placeholder,
  ariaLabel,
  testId,
}) => {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  const filteredOptions = useMemo(() => {
    const q = value.trim().toLowerCase();
    if (!q) return options;
    const exactMatch = options.some((opt) => opt.toLowerCase() === q);
    if (exactMatch) return options;
    return options.filter((opt) => opt.toLowerCase().includes(q));
  }, [options, value]);

  return (
    <div
      ref={wrapRef}
      className="combobox-wrap"
      onBlur={(e) => {
        if (!wrapRef.current?.contains(e.relatedTarget as Node | null)) {
          setOpen(false);
        }
      }}
    >
      <input
        type="text"
        placeholder={placeholder}
        aria-label={ariaLabel ?? placeholder}
        data-testid={testId}
        value={value}
        onFocus={() => setOpen(true)}
        onChange={(e) => {
          onChange(e.target.value);
          setOpen(true);
        }}
      />
      {open && options.length > 0 && (
        <div className="combobox-dropdown" role="listbox" aria-label={placeholder}>
          {value !== "" && (
            <button
              type="button"
              className="combobox-option muted"
              onMouseDown={(e) => {
                e.preventDefault();
                onChange("");
                setOpen(false);
              }}
            >
              All (clear filter)
            </button>
          )}
          {filteredOptions.length === 0 ? (
            <div className="small muted" style={{ padding: "4px 8px" }}>
              No matches
            </div>
          ) : (
            filteredOptions.map((opt) => (
              <button
                key={opt}
                type="button"
                role="option"
                aria-selected={value === opt}
                className={`combobox-option ${value === opt ? "active" : ""}`}
                onMouseDown={(e) => {
                  e.preventDefault();
                  onChange(opt);
                  setOpen(false);
                }}
              >
                {opt}
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
};
