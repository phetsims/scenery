// Copyright 2021-2023, University of Colorado Boulder

/**
 * Responsible for setting the provided focusProperty to null when the Focused node either
 * becomes invisible on the Display or is removed from the scene graph. It uses a
 * TrailVisibilityTracker to determine if any Node in the Trail has become invisible.
 *
 * Meant to be scenery-internal and used by FocusManager.
 *
 * @author Jesse Greenberg
 */

import TProperty from '../../../axon/js/TProperty.js';
import optionize from '../../../phet-core/js/optionize.js';
import { Focus, Instance, Node, scenery, TrailVisibilityTracker } from '../imports.js';

type FocusDisplayedControllerOptions = {

  // Extra work to do after the focusProperty is set to null because the focused Node is no longer displayed (it has
  // become invisible or has been removed from the scene graph).
  onRemoveFocus?: () => void;
};

class FocusDisplayedController {

  // last Node of the Trail that is focused, referenced so we can add and remove listeners from it
  private node: Node | null = null;

  // Observes the Trail to the Node and notifies when it has become invisible
  private visibilityTracker: TrailVisibilityTracker | null = null;

  // When there is value, we will watch and update when there are changes to the displayed state of the Focus trail.
  private focusProperty: TProperty<Focus | null> | null;
  private readonly onRemoveFocus: () => void;

  // Bound functions that are called when the displayed state of the Node changes.
  private readonly boundVisibilityListener: () => void;
  private readonly boundInstancesChangedListener: ( instance: Instance ) => void;

  // Handles changes to focus, adding or removing listeners
  private readonly boundFocusListener: ( focus: Focus | null ) => void;

  public constructor( focusProperty: TProperty<Focus | null>, providedOptions?: FocusDisplayedControllerOptions ) {

    const options = optionize<FocusDisplayedControllerOptions>()( {
      onRemoveFocus: _.noop
    }, providedOptions );
    assert && assert( typeof options.onRemoveFocus === 'function', 'invalid type for onRemoveFocus' );

    this.focusProperty = focusProperty;
    this.onRemoveFocus = options.onRemoveFocus;

    this.boundVisibilityListener = this.handleTrailVisibilityChange.bind( this );
    this.boundInstancesChangedListener = this.handleInstancesChange.bind( this );

    this.boundFocusListener = this.handleFocusChange.bind( this );
    this.focusProperty.link( this.boundFocusListener );
  }

  /**
   * When Focus changes, remove any listeners that were attached from last Focus and
   * add new listeners if focus has a new value.
   */
  private handleFocusChange( focus: Focus | null ): void {
    this.removeDisplayedListeners();

    if ( focus ) {
      this.addDisplayedListeners( focus );
    }
  }

  /**
   * When the Trail becomes invisible, Focus should be set to null.
   */
  private handleTrailVisibilityChange(): void {
    if ( this.visibilityTracker && !this.visibilityTracker.trailVisibleProperty.value ) {
      this.focusProperty!.value = null;
      this.onRemoveFocus();
    }
  }

  /**
   * If there are no more Instances for the Node with focus it has been removed from
   * the scene graph and so Focus should be set to null.
   */
  private handleInstancesChange( instance: Instance ): void {
    if ( instance.node && instance.node.instances.length === 0 ) {
      this.focusProperty!.value = null;
      this.onRemoveFocus();
    }
  }

  /**
   * Add listeners that watch when the Displayed state of the Node with Focus has changed,
   * including visibility of the trail and attachment to a scene graph.
   */
  private addDisplayedListeners( focus: Focus ): void {
    assert && assert( this.visibilityTracker === null, 'creating a new TrailVisibilityTracker but the last one was not disposed' );
    assert && assert( this.node === null, 'Still a reference to the previously focused Node, possible memory leak' );

    this.visibilityTracker = new TrailVisibilityTracker( focus.trail );
    this.visibilityTracker.addListener( this.boundVisibilityListener );

    this.node = focus.trail.lastNode();
    this.node.changedInstanceEmitter.addListener( this.boundInstancesChangedListener );
  }

  /**
   * Remove any listeners that were added to observables that fire when the Node's displayed
   * state may have changed.
   */
  private removeDisplayedListeners(): void {
    if ( this.visibilityTracker ) {
      this.visibilityTracker.removeListener( this.boundVisibilityListener );
      this.visibilityTracker.dispose();
      this.visibilityTracker = null;
    }
    if ( this.node ) {
      this.node.changedInstanceEmitter.removeListener( this.boundInstancesChangedListener );
      this.node = null;
    }
  }

  public dispose(): void {

    // this disposes the TrailVisibilityTracker and removes any listeners on the Node
    this.removeDisplayedListeners();
    this.focusProperty!.unlink( this.boundFocusListener );

    this.node = null;
    this.visibilityTracker = null;
    this.focusProperty = null;
  }
}

scenery.register( 'FocusDisplayedController', FocusDisplayedController );
export default FocusDisplayedController;