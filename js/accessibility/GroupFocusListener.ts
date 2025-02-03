// Copyright 2025, University of Colorado Boulder

/**
 * A listener that helps track focus for groups of Nodes. Properties allow you to observe when focus enters a group or
 * moves within a group of Nodes.
 *
 * Scenery (and native DOM) events make this difficult to track because the group receives focusin/focusout events
 * even when focus moves within the group. This listener handles that for you.
 *
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

import BooleanProperty from '../../../axon/js/BooleanProperty.js';
import Property from '../../../axon/js/Property.js';
import TProperty from '../../../axon/js/TProperty.js';
import Node from '../nodes/Node.js';
import scenery from '../scenery.js';
import SceneryEvent from '../input/SceneryEvent.js';
import type TInputListener from '../input/TInputListener.js';
import { pdomFocusProperty } from './pdomFocusProperty.js';

export default class GroupFocusListener implements TInputListener {

  // True when the focus is somewhere in this group.
  public readonly focusInGroupProperty: TProperty<boolean>;

  // The target of the focus, or null if the focus is not in this group.
  public readonly focusTargetProperty: TProperty<Node | null>;

  // True when the focus was in the group on the last focusout event. See the public focusWasInGroup getter.
  private _focusWasInGroup = false;

  private readonly groupParent: Node;

  public constructor( groupParent: Node ) {
    this.focusInGroupProperty = new BooleanProperty( false );
    this.focusTargetProperty = new Property( null );
    this.groupParent = groupParent;
  }

  public focusin( event: SceneryEvent ): void {
    this.focusInGroupProperty.value = true;
    this.focusTargetProperty.value = event.target;
    this._focusWasInGroup = true;
  }

  public focusout( event: SceneryEvent ): void {
    const nextTargetTrail = pdomFocusProperty.value?.trail;
    if ( nextTargetTrail && nextTargetTrail.containsNode( this.groupParent ) ) {

      // The focusTargetProperty will be updated in the focusin event so there is
      // nothing to do
    }
    else {
      this.focusInGroupProperty.value = false;
      this.focusTargetProperty.value = null;
      this._focusWasInGroup = false;
    }
  }

  /**
   * A useful Property to check whether focus was already in the group when the focusTargetProperty changes.
   *
   * Example usage:
   *
   * groupFocusListener.focusTargetProperty.link( focusTarget => {
   *   if ( focusTarget ) {
   *     if ( groupFocusListener.focusWasInGroup ) {
   *       // Focus moved within the group but did not leave the group since the last focus event.
   *     }
   *     else {
   *       // Focus just entered the group.
   *     }
   *   }
   * } );
   */
  public get focusWasInGroup(): boolean {
    return this._focusWasInGroup;
  }

  public dispose(): void {
    this.focusInGroupProperty.dispose();
    this.focusTargetProperty.dispose();
  }
}

scenery.register( 'GroupFocusListener', GroupFocusListener );