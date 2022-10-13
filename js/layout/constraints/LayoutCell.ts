// Copyright 2022, University of Colorado Boulder

/**
 * A configurable cell containing a Node used for more permanent layouts
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Orientation from '../../../../phet-core/js/Orientation.js';
import { LayoutConstraint, LayoutProxy, LayoutProxyProperty, Node, scenery } from '../../imports.js';

// NOTE: This would be an abstract class, but that is incompatible with how mixin constraints work in TypeScript
export default class LayoutCell {

  // We might need to notify the constraint it needs a layout
  private readonly _constraint: LayoutConstraint;

  private readonly _node: Node;

  // Our proxy will be dynamically computed and updated (based on whether there is a valid ancestorNode=>node trail)
  // Generally used to compute layout in the node's parent coordinate frame.
  private _proxy: LayoutProxy | null;

  // Called when layoutOptions changes for our Node
  private readonly layoutOptionsListener: () => void;

  // If we're not provided a (static) LayoutProxy in our constructor, we'll track and generate LayoutProxies with this.
  private readonly layoutProxyProperty: LayoutProxyProperty | null;

  /**
   * NOTE: Consider this scenery-internal AND protected. It's effectively a protected constructor for an abstract type,
   * but cannot be due to how mixins constrain things (TypeScript doesn't work with private/protected things like this)
   *
   * NOTE: Methods can be marked as protected, however!
   *
   * (scenery-internal)
   *
   * @param constraint
   * @param node
   * @param proxy - If not provided, LayoutProxies will be computed and updated based on the ancestorNode of the
   *                constraint. This includes more work, and ideally should be avoided for things like FlowBox/GridBox
   *                (but will be needed by ManualConstraint or other direct LayoutConstraint usage)
   */
  public constructor( constraint: LayoutConstraint, node: Node, proxy: LayoutProxy | null ) {
    if ( proxy ) {
      this.layoutProxyProperty = null;
      this._proxy = proxy;
    }
    else {

      this._proxy = null;

      // If a LayoutProxy is not provided, we'll listen to (a) all the trails between our ancestor and this node,
      // (b) construct layout proxies for it (and assign here), and (c) listen to ancestor transforms to refresh
      // the layout when needed.
      this.layoutProxyProperty = new LayoutProxyProperty( constraint.ancestorNode, node, {
        onTransformChange: () => constraint.updateLayoutAutomatically()
      } );
      this.layoutProxyProperty.link( proxy => {
        this._proxy = proxy;

        constraint.updateLayoutAutomatically();
      } );
    }

    this._constraint = constraint;
    this._node = node;

    this.layoutOptionsListener = this.onLayoutOptionsChange.bind( this );
    this.node.layoutOptionsChangedEmitter.addListener( this.layoutOptionsListener );
  }

  // Can't be abstract, we're using mixins :(
  protected onLayoutOptionsChange(): void {
    // Lint rule not needed here
  }

  /**
   * (scenery-internal)
   */
  public get node(): Node {
    return this._node;
  }

  /**
   * (scenery-internal)
   */
  public isConnected(): boolean {
    return this._proxy !== null;
  }

  /**
   * (scenery-internal)
   */
  public get proxy(): LayoutProxy {
    assert && assert( this._proxy );

    return this._proxy!;
  }

  /**
   * (scenery-internal)
   */
  public isSizable( orientation: Orientation ): boolean {
    return orientation === Orientation.HORIZONTAL ? this.proxy.widthSizable : this.proxy.heightSizable;
  }

  /**
   * Releases references
   */
  public dispose(): void {
    this.layoutProxyProperty && this.layoutProxyProperty.dispose();

    this.node.layoutOptionsChangedEmitter.removeListener( this.layoutOptionsListener );
  }
}

scenery.register( 'LayoutCell', LayoutCell );
