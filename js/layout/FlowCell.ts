// Copyright 2021-2022, University of Colorado Boulder

/**
 * A configurable cell containing a Node used for FlowConstraint layout
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import Utils from '../../../dot/js/Utils.js';
import Orientation from '../../../phet-core/js/Orientation.js';
import { FlowConfigurable, FlowConfigurableAlign, FlowConfigurableOptions, FlowConstraint, LayoutProxy, Node, scenery } from '../imports.js';
import LayoutCell from './LayoutCell.js';

export default class FlowCell extends FlowConfigurable( LayoutCell ) {

  _pendingSize: number; // scenery-internal
  private _flowConstraint: FlowConstraint;

  constructor( constraint: FlowConstraint, node: Node, proxy: LayoutProxy | null ) {
    super( constraint, node, proxy );

    this._flowConstraint = constraint;
    this._pendingSize = 0;

    this.orientation = constraint.orientation;
    this.onLayoutOptionsChange();
  }

  get effectiveAlign(): FlowConfigurableAlign {
    return this._align !== null ? this._align : this._flowConstraint._align!;
  }

  get effectiveStretch(): boolean {
    return this._stretch !== null ? this._stretch : this._flowConstraint._stretch!;
  }

  get effectiveLeftMargin(): number {
    return this._leftMargin !== null ? this._leftMargin : this._flowConstraint._leftMargin!;
  }


  get effectiveRightMargin(): number {
    return this._rightMargin !== null ? this._rightMargin : this._flowConstraint._rightMargin!;
  }


  get effectiveTopMargin(): number {
    return this._topMargin !== null ? this._topMargin : this._flowConstraint._topMargin!;
  }


  get effectiveBottomMargin(): number {
    return this._bottomMargin !== null ? this._bottomMargin : this._flowConstraint._bottomMargin!;
  }


  get effectiveGrow(): number {
    return this._grow !== null ? this._grow : this._flowConstraint._grow!;
  }

  get effectiveMinContentWidth(): number | null {
    return this._minContentWidth !== null ? this._minContentWidth : this._flowConstraint._minContentWidth;
  }

  get effectiveMinContentHeight(): number | null {
    return this._minContentHeight !== null ? this._minContentHeight : this._flowConstraint._minContentHeight;
  }

  get effectiveMaxContentWidth(): number | null {
    return this._maxContentWidth !== null ? this._maxContentWidth : this._flowConstraint._maxContentWidth;
  }

  get effectiveMaxContentHeight(): number | null {
    return this._maxContentHeight !== null ? this._maxContentHeight : this._flowConstraint._maxContentHeight;
  }

  protected override onLayoutOptionsChange(): void {
    if ( this.node.layoutOptions ) {
      this.setOptions( this.node.layoutOptions as FlowConfigurableOptions );
    }

    super.onLayoutOptionsChange();
  }

  private setOptions( options?: FlowConfigurableOptions ): void {
    this.setConfigToInherit();
    this.mutateConfigurable( options );
  }

  getMinimumSize( orientation: Orientation ): number {
    if ( orientation === Orientation.HORIZONTAL ) {
      return this.effectiveLeftMargin +
             Math.max( this.proxy.minimumWidth, this.effectiveMinContentWidth || 0 ) +
             this.effectiveRightMargin;
    }
    else {
      return this.effectiveTopMargin +
             Math.max( this.proxy.minimumHeight, this.effectiveMinContentHeight || 0 ) +
             this.effectiveBottomMargin;
    }
  }

  getMaximumSize( orientation: Orientation ): number {
    const isSizable = orientation === Orientation.HORIZONTAL ? this.proxy.widthSizable : this.proxy.heightSizable;

    if ( orientation === Orientation.HORIZONTAL ) {
      return this.effectiveLeftMargin +
             Math.min( isSizable ? Number.POSITIVE_INFINITY : this.proxy.width, this.effectiveMaxContentWidth || Number.POSITIVE_INFINITY ) +
             this.effectiveRightMargin;
    }
    else {
      return this.effectiveTopMargin +
             Math.min( isSizable ? Number.POSITIVE_INFINITY : this.proxy.height, this.effectiveMaxContentHeight || Number.POSITIVE_INFINITY ) +
             this.effectiveBottomMargin;
    }
  }

  attemptPreferredSize( orientation: Orientation, value: number ): void {
    if ( this.isSizable( orientation ) ) {
      const minimumSize = this.getMinimumSize( orientation );
      const maximumSize = this.getMaximumSize( orientation );

      assert && assert( isFinite( minimumSize ) );
      assert && assert( maximumSize >= minimumSize );

      value = Utils.clamp( value, minimumSize, maximumSize );

      if ( orientation === Orientation.HORIZONTAL ) {
        this.proxy.preferredWidth = value - this.effectiveLeftMargin - this.effectiveRightMargin;
      }
      else {
        this.proxy.preferredHeight = value - this.effectiveTopMargin - this.effectiveBottomMargin;
      }
      // TODO: warnings if those preferred sizes weren't reached?
    }
  }

  positionStart( orientation: Orientation, value: number ): void {
    if ( orientation === Orientation.HORIZONTAL ) {
      const left = this.effectiveLeftMargin + value;

      if ( Math.abs( this.proxy.left - left ) > 1e-9 ) {
        this.proxy.left = left;
      }
    }
    else {
      const top = this.effectiveTopMargin + value;

      if ( Math.abs( this.proxy.top - top ) > 1e-9 ) {
        this.proxy.top = top;
      }
    }
  }

  positionOrigin( orientation: Orientation, value: number ): void {
    if ( orientation === Orientation.HORIZONTAL ) {
      if ( Math.abs( this.proxy.x - value ) > 1e-9 ) {
        this.proxy.x = value;
      }
    }
    else {
      if ( Math.abs( this.proxy.y - value ) > 1e-9 ) {
        this.proxy.y = value;
      }
    }
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

scenery.register( 'FlowCell', FlowCell );
