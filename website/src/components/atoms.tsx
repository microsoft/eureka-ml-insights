import React from "react";
import { Modal, Table, Tooltip, theme } from "antd";

export const ControlRowView = ({
    title,
    description,
    value,
    control,
    className,
  }: {
    title: string;
    description: string;
    value: string | number | boolean;
    control: any;
    className?: string;
    truncateLength?: number;
  }) => {
    return (
      <div className={`${className}`}>
        <div>
          <span className="text-primary inline-block">{title} </span>
          <span className="text-xs ml-1 text-accent -mt-2 inline-block">
            {value}
          </span>{" "}
          <Tooltip title={description}>
            
          </Tooltip>
        </div>
        {control}
        <div className="bordper-b  border-secondary border-dashed pb-2 mxp-2"></div>
      </div>
    );
  };

  export const CheckBoxView = ({
    title,
    description,
    value,
    control,
    className,
  }: {
    title: string;
    description: string;
    value: string | number | boolean;
    control: any;
    className?: string;
    truncateLength?: number;
  }) => {
    return (
      <div className={`${className}`}>
        <div>
          <span className="text-primary inline-block">{title} </span>
          <Checkbox indeterminate={indeterminate} onChange={onCheckAllChange} checked={checkAll}>
            Check all
          </Checkbox>
          <Divider />
          <CheckboxGroup options={plainOptions} value={checkedList} onChange={onChange} />
          <Tooltip title={description}>
            
          </Tooltip>
        </div>
        {control}
        <div className="bordper-b  border-secondary border-dashed pb-2 mxp-2"></div>
      </div>
    );
  };