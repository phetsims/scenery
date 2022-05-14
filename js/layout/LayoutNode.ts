// Copyright 2022, University of Colorado Boulder

/**
 * Supertype for layout Nodes
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { HeightSizable, HeightSizableSelfOptions, Node, NodeLayoutConstraint, NodeOptions, scenery, WidthSizable, WidthSizableSelfOptions } from '../imports.js';

type SelfOptions = {
  // Controls whether the layout container will re-trigger layout automatically after the "first" layout during
  // construction. The layout container will layout once after processing the options object, but if resize:false,
  // then after that manual layout calls will need to be done (with updateLayout())
  resize?: boolean;
};

export const LAYOUT_NODE_OPTION_KEYS = [ 'resize' ] as const;

type SuperOptions = NodeOptions & WidthSizableSelfOptions & HeightSizableSelfOptions;

export type LayoutNodeOptions = SelfOptions & SuperOptions;

export default abstract class LayoutNode<Constraint extends NodeLayoutConstraint> extends WidthSizable( HeightSizable( Node ) ) {

  protected _constraint!: Constraint;

  constructor( providedOptions?: LayoutNodeOptions ) {
    super( providedOptions );
  }

  protected linkLayoutBounds(): void {
    // Adjust the localBounds to be the laid-out area
    this._constraint.layoutBoundsProperty.link( layoutBounds => {
      this.localBounds = layoutBounds;
    } );
  }

  override setExcludeInvisibleChildrenFromBounds( excludeInvisibleChildrenFromBounds: boolean ): void {
    super.setExcludeInvisibleChildrenFromBounds( excludeInvisibleChildrenFromBounds );

    this._constraint.excludeInvisible = excludeInvisibleChildrenFromBounds;
  }

  override setChildren( children: Node[] ): this {

    // If the layout is already locked, we need to bail and only call Node's setChildren.
    if ( this.constraint.isLocked ) {
      return super.setChildren( children );
    }

    const oldChildren = this.getChildren(); // defensive copy

    // Lock layout while the children are removed and added
    this.constraint.lock();
    super.setChildren( children );
    this.constraint.unlock();

    // Determine if the children array has changed. We'll gain a performance benefit by not triggering layout when
    // the children haven't changed.
    if ( !_.isEqual( oldChildren, children ) ) {
      this.constraint.updateLayoutAutomatically();
    }

    return this;
  }

  /**
   * Manually run the layout (for instance, if resize:false is currently set, or if there is other hackery going on).
   */
  updateLayout(): void {
    this._constraint.updateLayout();
  }

  get resize(): boolean {
    return this._constraint.enabled;
  }

  set resize( value: boolean ) {
    this._constraint.enabled = value;
  }

  /**
   * Manual access to the constraint
   */
  get constraint(): Constraint {
    return this._constraint;
  }
}

scenery.register( 'LayoutNode', LayoutNode );
