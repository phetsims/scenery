// Copyright 2021-2022, University of Colorado Boulder

/**
 * A configurable cell containing a Node used for FlowConstraint layout
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Bounds2 from '../../../dot/js/Bounds2.js';
import Utils from '../../../dot/js/Utils.js';
import Orientation from '../../../phet-core/js/Orientation.js';
import { FlowConfigurable, FlowConfigurableAlign, FlowConfigurableOptions, FlowConstraint, LayoutProxy, scenery, Node, LayoutProxyProperty, TransformTracker } from '../imports.js';

export default class FlowCell extends FlowConfigurable( Object ) {

  private readonly _constraint: FlowConstraint;
  private readonly _node: Node;
  private _proxy: LayoutProxy | null;
  _pendingSize: number; // scenery-internal
  private readonly layoutOptionsListener: () => void;
  private readonly layoutProxyProperty: LayoutProxyProperty | null;
  private transformTracker: TransformTracker | null;

  constructor( constraint: FlowConstraint, node: Node, proxy: LayoutProxy | null ) {
    super();

    this.transformTracker = null;

    if ( proxy ) {
      this.layoutProxyProperty = null;
    }
    else {
      // If a LayoutProxy is not provided, we'll listen to (a) all the trails between our ancestor and this node,
      // (b) construct layout proxies for it (and assign here), and (c) listen to ancestor transforms to refresh
      // the layout when needed.
      this.layoutProxyProperty = new LayoutProxyProperty( constraint.ancestorNode, node );
      this.layoutProxyProperty.link( proxy => {
        this._proxy = proxy;
        if ( this.transformTracker ) {
          this.transformTracker.dispose();
        }
        if ( proxy ) {
          this.transformTracker = new TransformTracker( proxy.trail!.copy().addAncestor( constraint.ancestorNode ) );
          this.transformTracker.addListener( () => constraint.updateLayoutAutomatically() );
        }
      } );
    }

    this._constraint = constraint;
    this._node = node;
    this._proxy = constraint.createLayoutProxy( node )!; // TODO: handle disconnected, and listen for if we disconnect
    this._pendingSize = 0;

    this.orientation = constraint.orientation;
    this.onLayoutOptionsChange();

    this.layoutOptionsListener = this.onLayoutOptionsChange.bind( this );
    this.node.layoutOptionsChangedEmitter.addListener( this.layoutOptionsListener );
  }

  get effectiveAlign(): FlowConfigurableAlign {
    return this._align !== null ? this._align : this._constraint._align!;
  }

  get effectiveStretch(): boolean {
    return this._stretch !== null ? this._stretch : this._constraint._stretch!;
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

  isConnected(): boolean {
    return this._proxy !== null;
  }

  get proxy(): LayoutProxy {
    assert && assert( this._proxy );

    return this._proxy!;
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

  isSizable( orientation: Orientation ): boolean {
    return orientation === Orientation.HORIZONTAL ? this.proxy.widthSizable : this.proxy.heightSizable;
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

  /**
   * Releases references
   */
  dispose(): void {
    this.layoutProxyProperty && this.layoutProxyProperty.dispose();
    this.transformTracker && this.transformTracker.dispose();

    this.node.layoutOptionsChangedEmitter.removeListener( this.layoutOptionsListener );
  }
}

scenery.register( 'FlowCell', FlowCell );
