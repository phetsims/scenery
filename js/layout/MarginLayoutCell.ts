// Copyright 2021-2022, University of Colorado Boulder

/**
 * A LayoutCell that has margins, and can be positioned and sized relative to those.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import Utils from '../../../dot/js/Utils.js';
import Orientation from '../../../phet-core/js/Orientation.js';
import OrientationPair from '../../../phet-core/js/OrientationPair.js';
import { LayoutCell, LayoutConstraint, LayoutProxy, Node, scenery } from '../imports.js';

const sizableFlagPair = new OrientationPair( 'widthSizable' as const, 'heightSizable' as const );
const preferredSizePair = new OrientationPair( 'preferredWidth' as const, 'preferredHeight' as const );

// Position changes smaller than this will be ignored
const CHANGE_POSITION_THRESHOLD = 1e-9;

// Interface expected to be overridden by subtypes (GridCell, FlowCell)
export interface MarginLayout {
  _leftMargin: number | null;
  _rightMargin: number | null;
  _topMargin: number | null;
  _bottomMargin: number | null;
  _minContentWidth: number | null;
  _minContentHeight: number | null;
  _maxContentWidth: number | null;
  _maxContentHeight: number | null;
}

export type MarginLayoutConstraint = LayoutConstraint & MarginLayout;

export default class MarginLayoutCell extends LayoutCell {

  private _marginConstraint: MarginLayoutConstraint;

  // These will get overridden, they're needed since mixins have many limitations and we'd have to have a ton of casts
  // without these existing.
  _leftMargin!: number | null;
  _rightMargin!: number | null;
  _topMargin!: number | null;
  _bottomMargin!: number | null;
  _minContentWidth!: number | null;
  _minContentHeight!: number | null;
  _maxContentWidth!: number | null;
  _maxContentHeight!: number | null;

  constructor( constraint: MarginLayoutConstraint, node: Node, proxy: LayoutProxy | null ) {
    super( constraint, node, proxy );

    this._marginConstraint = constraint;
  }

  get effectiveLeftMargin(): number {
    return this._leftMargin !== null ? this._leftMargin : this._marginConstraint._leftMargin!;
  }

  get effectiveRightMargin(): number {
    return this._rightMargin !== null ? this._rightMargin : this._marginConstraint._rightMargin!;
  }

  get effectiveTopMargin(): number {
    return this._topMargin !== null ? this._topMargin : this._marginConstraint._topMargin!;
  }

  get effectiveBottomMargin(): number {
    return this._bottomMargin !== null ? this._bottomMargin : this._marginConstraint._bottomMargin!;
  }

  getEffectiveMinMargin( orientation: Orientation ): number {
    return orientation === Orientation.HORIZONTAL ? this.effectiveLeftMargin : this.effectiveTopMargin;
  }

  getEffectiveMaxMargin( orientation: Orientation ): number {
    return orientation === Orientation.HORIZONTAL ? this.effectiveRightMargin : this.effectiveBottomMargin;
  }

  get effectiveMinContentWidth(): number | null {
    return this._minContentWidth !== null ? this._minContentWidth : this._marginConstraint._minContentWidth;
  }

  get effectiveMinContentHeight(): number | null {
    return this._minContentHeight !== null ? this._minContentHeight : this._marginConstraint._minContentHeight;
  }

  getEffectiveMinContent( orientation: Orientation ): number | null {
    return orientation === Orientation.HORIZONTAL ? this.effectiveMinContentWidth : this.effectiveMinContentHeight;
  }

  get effectiveMaxContentWidth(): number | null {
    return this._maxContentWidth !== null ? this._maxContentWidth : this._marginConstraint._maxContentWidth;
  }

  get effectiveMaxContentHeight(): number | null {
    return this._maxContentHeight !== null ? this._maxContentHeight : this._marginConstraint._maxContentHeight;
  }

  getEffectiveMaxContent( orientation: Orientation ): number | null {
    return orientation === Orientation.HORIZONTAL ? this.effectiveMaxContentWidth : this.effectiveMaxContentHeight;
  }

  getMinimumSize( orientation: Orientation ): number {
    return this.getEffectiveMinMargin( orientation ) +
           Math.max( this.proxy.getMinimum( orientation ), this.getEffectiveMinContent( orientation ) || 0 ) +
           this.getEffectiveMaxMargin( orientation );
  }

  getMaximumSize( orientation: Orientation ): number {
    return this.getEffectiveMinMargin( orientation ) +
           ( this.getEffectiveMaxContent( orientation ) || Number.POSITIVE_INFINITY ) +
           this.getEffectiveMaxMargin( orientation );
  }

  attemptPreferredSize( orientation: Orientation, value: number ): void {
    if ( this.proxy[ sizableFlagPair.get( orientation ) ] ) {
      const minimumSize = this.getMinimumSize( orientation );
      const maximumSize = this.getMaximumSize( orientation );

      assert && assert( isFinite( minimumSize ) );
      assert && assert( maximumSize >= minimumSize );

      value = Utils.clamp( value, minimumSize, maximumSize );

      this.proxy[ preferredSizePair.get( orientation ) ] = value - this.getEffectiveMinMargin( orientation ) - this.getEffectiveMaxMargin( orientation );
    }
  }

  positionStart( orientation: Orientation, value: number ): void {
    const start = this.getEffectiveMinMargin( orientation ) + value;

    if ( Math.abs( this.proxy[ orientation.minSide ] - start ) > CHANGE_POSITION_THRESHOLD ) {
      this.proxy[ orientation.minSide ] = start;
    }
  }

  positionOrigin( orientation: Orientation, value: number ): void {
    if ( Math.abs( this.proxy[ orientation.coordinate ] - value ) > CHANGE_POSITION_THRESHOLD ) {
      this.proxy[ orientation.coordinate ] = value;
    }
  }

  /**
   * Returns the bounding box of the cell if it was repositioned to have its origin shifted to the origin of the
   * ancestor node's local coordinate frame.
   */
  getOriginBounds(): Bounds2 {
    return this.getCellBounds().shiftedXY( -this.proxy.x, -this.proxy.y );
  }

  getCellBounds(): Bounds2 {
    return this.proxy.bounds.withOffsets(
      this.effectiveLeftMargin,
      this.effectiveTopMargin,
      this.effectiveRightMargin,
      this.effectiveBottomMargin
    );
  }
}

scenery.register( 'MarginLayoutCell', MarginLayoutCell );
