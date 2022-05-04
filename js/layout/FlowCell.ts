// Copyright 2021-2022, University of Colorado Boulder

/**
 * A configurable cell containing a Node used for FlowConstraint layout
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import Utils from '../../../dot/js/Utils.js';
import Orientation from '../../../phet-core/js/Orientation.js';
import { FlowConfigurable, FlowConfigurableAlign, FlowConfigurableOptions, FlowConstraint, LayoutProxy, scenery, Node } from '../imports.js';

export default class FlowCell extends FlowConfigurable( Object ) {

  private _constraint: FlowConstraint;
  private _node: Node;
  private _proxy: LayoutProxy;
  public _pendingSize: number; // scenery-internal
  private layoutOptionsListener: () => void;

  constructor( constraint: FlowConstraint, node: Node ) {
    super();

    this._constraint = constraint;
    this._node = node;
    this._proxy = constraint.createLayoutProxy( node );
    this._pendingSize = 0;

    this.onLayoutOptionsChange();

    this.layoutOptionsListener = this.onLayoutOptionsChange.bind( this );
    this.node.layoutOptionsChangedEmitter.addListener( this.layoutOptionsListener );
  }

  get effectiveAlign(): FlowConfigurableAlign {
    return this._align !== null ? this._align : this._constraint._align!;
  }

  get effectiveLeftMargin(): number {
    return this._leftMargin !== null ? this._leftMargin : this._constraint._leftMargin!;
  }


  get effectiveRightMargin(): number {
    return this._rightMargin !== null ? this._rightMargin : this._constraint._rightMargin!;
  }


  get effectiveTopMargin(): number {
    return this._topMargin !== null ? this._topMargin : this._constraint._topMargin!;
  }


  get effectiveBottomMargin(): number {
    return this._bottomMargin !== null ? this._bottomMargin : this._constraint._bottomMargin!;
  }


  get effectiveGrow(): number {
    return this._grow !== null ? this._grow : this._constraint._grow!;
  }

  get effectiveMinContentWidth(): number | null {
    return this._minContentWidth !== null ? this._minContentWidth : this._constraint._minContentWidth;
  }

  get effectiveMinContentHeight(): number | null {
    return this._minContentHeight !== null ? this._minContentHeight : this._constraint._minContentHeight;
  }

  get effectiveMaxContentWidth(): number | null {
    return this._maxContentWidth !== null ? this._maxContentWidth : this._constraint._maxContentWidth;
  }

  get effectiveMaxContentHeight(): number | null {
    return this._maxContentHeight !== null ? this._maxContentHeight : this._constraint._maxContentHeight;
  }

  private onLayoutOptionsChange(): void {
    if ( this.node.layoutOptions ) {
      this.setOptions( this.node.layoutOptions as FlowConfigurableOptions );
    }
  }

  private setOptions( options?: FlowConfigurableOptions ): void {
    this.setConfigToInherit();
    this.mutateConfigurable( options );
  }

  get node(): Node {
    return this._node;
  }

  get proxy(): LayoutProxy {
    return this._proxy;
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
    if ( orientation === Orientation.HORIZONTAL ? this.proxy.widthSizable : this.proxy.heightSizable ) {
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

  /**
   * Releases references
   */
  dispose(): void {
    this.node.layoutOptionsChangedEmitter.removeListener( this.layoutOptionsListener );
  }
}

scenery.register( 'FlowCell', FlowCell );
