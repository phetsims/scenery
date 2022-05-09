// Copyright 2021-2022, University of Colorado Boulder

import { FlowHorizontalAlign, FlowOrientation, FlowVerticalAlign, GridHorizontalAlign, GridVerticalAlign } from '../imports.js';

/**
 * The main type interface for layout options
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

export default interface ILayoutOptions {
  orientation?: FlowOrientation;
  align?: FlowHorizontalAlign | FlowVerticalAlign;
  stretch?: boolean;
  xAlign?: GridHorizontalAlign;
  yAlign?: GridVerticalAlign;
  grow?: number;
  xGrow?: number;
  yGrow?: number;
  margin?: number;
  xMargin?: number;
  yMargin?: number;
  leftMargin?: number;
  rightMargin?: number;
  topMargin?: number;
  bottomMargin?: number;
  minContentWidth?: number;
  minContentHeight?: number;
  maxContentWidth?: number;
  maxContentHeight?: number;
  wrap?: boolean;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
} // eslint-disable-line
