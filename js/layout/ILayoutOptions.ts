// Copyright 2021, University of Colorado Boulder

/**
 * The main type interface for layout options
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

interface ILayoutOptions {
  orientation?: string,
  align?: string,
  xAlign?: string,
  yAlign?: string,
  grow?: number,
  xGrow?: number,
  yGrow?: number,
  margin?: number,
  xMargin?: number,
  yMargin?: number,
  leftMargin?: number,
  rightMargin?: number,
  topMargin?: number,
  bottomMargin?: number,
  minContentWidth?: number,
  minContentHeight?: number,
  maxContentWidth?: number,
  maxContentHeight?: number
}

export default ILayoutOptions;