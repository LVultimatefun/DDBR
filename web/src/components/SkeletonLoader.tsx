import React from "react";
import { useI18n } from "../stores/i18nStore";

export interface TableSkeletonProps {
  rows?: number;
  columns?: number;
}

export const TableSkeleton: React.FC<TableSkeletonProps> = ({ rows = 6, columns = 8 }) => {
  const { t } = useI18n();
  return (
    <>
      {Array.from({ length: rows }).map((_, rowIdx) => (
        <tr
          key={`skel-row-${rowIdx}`}
          className="skeleton-table-row"
          aria-busy="true"
          data-testid={rowIdx === 0 ? "table-skeleton" : undefined}
        >
          {Array.from({ length: columns }).map((__, colIdx) => (
            <td key={`skel-cell-${rowIdx}-${colIdx}`}>
              <div
                className="skeleton-shimmer skeleton-line"
                style={{
                  width: colIdx === 0 ? "70%" : colIdx === 1 ? "85%" : "60%",
                }}
                aria-label={rowIdx === 0 && colIdx === 0 ? t("skeleton.loadingTable") : undefined}
              />
            </td>
          ))}
        </tr>
      ))}
    </>
  );
};

export const StudioSkeleton: React.FC = () => {
  const { t } = useI18n();
  return (
    <div
      className="studio-skeleton"
      role="status"
      aria-busy="true"
      aria-label={t("skeleton.loadingStudio")}
      data-testid="studio-skeleton"
    >
      <div className="skeleton-shimmer skeleton-bar-header" />
      <div className="studio-skeleton-grid">
        <div className="studio-skeleton-left">
          <div className="skeleton-shimmer skeleton-viewport" />
          <div className="skeleton-shimmer skeleton-timeline" />
        </div>
        <div className="studio-skeleton-right">
          <div className="skeleton-shimmer skeleton-inspector" />
        </div>
      </div>
    </div>
  );
};

export const PageSkeleton: React.FC = () => {
  const { t } = useI18n();
  return (
    <div
      className="page-container page-skeleton"
      role="status"
      aria-busy="true"
      aria-label={t("skeleton.loadingPage")}
      data-testid="page-skeleton"
    >
      <div className="skeleton-shimmer skeleton-page-title" />
      <div className="skeleton-shimmer skeleton-page-toolbar" />
      <div className="skeleton-shimmer skeleton-page-panel" />
    </div>
  );
};
