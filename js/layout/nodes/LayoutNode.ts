// Copyright 2022, University of Colorado Boulder

/**
 * Supertype for layout Nodes
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import IProperty from '../../../../axon/js/IProperty.js';
import Vector2 from '../../../../dot/js/Vector2.js';
import Vector2Property from '../../../../dot/js/Vector2Property.js';
import { Node, NodeLayoutConstraint, NodeOptions, scenery, Sizable, SizableOptions } from '../../imports.js';

type SelfOptions = {
  // Controls whether the layout container will re-trigger layout automatically after the "first" layout during
  // construction. The layout container will layout once after processing the options object, but if resize:false,
  // then after that manual layout calls will need to be done (with updateLayout())
  resize?: boolean;

  // Controls where the origin of the "layout" is placed (usually within the Node itself). For typical usages, this will
  // be (0,0) and the upper-left of the content will be placed there. `layoutOrigin` will adjust this point.
  // NOTE: If there is origin-based content, that content will be placed at this origin (and may go to the top/left of
  // this layoutOrigin).
  layoutOrigin?: Vector2;
};

export const LAYOUT_NODE_OPTION_KEYS = [ 'resize', 'layoutOrigin' ] as const;

type SuperOptions = NodeOptions & SizableOptions;

export type LayoutNodeOptions = SelfOptions & SuperOptions;

export default abstract class LayoutNode<Constraint extends NodeLayoutConstraint> extends Sizable( Node ) {

  protected _constraint!: Constraint;
  readonly layoutOriginProperty: IProperty<Vector2> = new Vector2Property( Vector2.ZERO );

  constructor( providedOptions?: LayoutNodeOptions ) {
    super();

    this.mutate( providedOptions );
  }

  protected linkLayoutBounds(): void {
    // Adjust the localBounds to be the laid-out area (defined by the constraint)
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

  get layoutOrigin(): Vector2 {
    return this.layoutOriginProperty.value;
  }

  set layoutOrigin( value: Vector2 ) {
    this.layoutOriginProperty.value = value;
  }

  /**
   * Manual access to the constraint
   */
  get constraint(): Constraint {
    return this._constraint;
  }

  /**
   * Releases references
   */
  override dispose(): void {
    this._constraint.dispose();

    super.dispose();
  }
}

scenery.register( 'LayoutNode', LayoutNode );