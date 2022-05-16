// Copyright 2022, University of Colorado Boulder

/**
 * A configurable cell containing a Node used for more permanent layouts
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Orientation from '../../../../phet-core/js/Orientation.js';
import { LayoutConstraint, LayoutProxy, Node, scenery, TrackingLayoutProxyProperty } from '../../imports.js';

export default class LayoutCell {

  private readonly _constraint: LayoutConstraint;
  private readonly _node: Node;
  private _proxy: LayoutProxy | null;
  private readonly layoutOptionsListener: () => void;
  private readonly layoutProxyProperty: TrackingLayoutProxyProperty | null;

  constructor( constraint: LayoutConstraint, node: Node, proxy: LayoutProxy | null ) {
    if ( proxy ) {
      this.layoutProxyProperty = null;
      this._proxy = proxy;
    }
    else {

      this._proxy = null;

      // If a LayoutProxy is not provided, we'll listen to (a) all the trails between our ancestor and this node,
      // (b) construct layout proxies for it (and assign here), and (c) listen to ancestor transforms to refresh
      // the layout when needed.
      this.layoutProxyProperty = new TrackingLayoutProxyProperty( constraint.ancestorNode, node, () => constraint.updateLayoutAutomatically() );
      this.layoutProxyProperty.link( proxy => {
        this._proxy = proxy;
      } );
    }

    this._constraint = constraint;
    this._node = node;

    this.layoutOptionsListener = this.onLayoutOptionsChange.bind( this );
    this.node.layoutOptionsChangedEmitter.addListener( this.layoutOptionsListener );
  }

  // Can't be abstract, we're using mixins :(
  protected onLayoutOptionsChange(): void {

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

  isSizable( orientation: Orientation ): boolean {
    return orientation === Orientation.HORIZONTAL ? this.proxy.widthSizable : this.proxy.heightSizable;
  }

  /**
   * Releases references
   */
  dispose(): void {
    this.layoutProxyProperty && this.layoutProxyProperty.dispose();

    this.node.layoutOptionsChangedEmitter.removeListener( this.layoutOptionsListener );
  }
}

scenery.register( 'LayoutCell', LayoutCell );
