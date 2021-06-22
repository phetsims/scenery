// Copyright 2021, University of Colorado Boulder

/**
 * Responsible for setting the provided focusProperty to null when the Focused node either
 * becomes invisible on the Display or is removed from the scene graph. It uses a
 * TrailVisibilityTracker to determine if any Node in the Trail has become invisible.
 *
 * Meant to be scenery-internal and used by FocusManager.
 *
 * @author Jesse Greenberg
 */

import merge from '../../../phet-core/js/merge.js';
import scenery from '../scenery.js';
import TrailVisibilityTracker from '../util/TrailVisibilityTracker.js';

class FocusDisplayedController {

  /**
   * @param {Property.<Focus>} focusProperty
   * @param {Object} [options]
   */
  constructor( focusProperty, options ) {

    options = merge( {

      // @function - Extra work to do after the focusProperty is set to null because
      // the focused Node is no longer displayed (it has become invisible or has
      // been removed from the scene graph).
      onRemoveFocus: () => {}
    }, options );
    assert && assert( typeof options.onRemoveFocus === 'function', 'invalid type for onRemoveFocus' );

    // @private {Node|null} - last Node of the Trail that is focused, referenced so we
    // can add and remove listeners from it
    this.node = null;

    // @private {TrailVisibilityTracker|null} - Observables the Trail to the Node and
    // notifies when it has become invisible
    this.visibilityTracker = null;

    // @private} {Property.<Focus>}
    this.focusProperty = focusProperty;

    // @private {function}
    this.onRemoveFocus = options.onRemoveFocus;

    // @private {function} - Bound functions that are called when the displayed state
    // of the Node changes.
    this.boundVisibilityListener = this.handleTrailVisibilityChange.bind( this );
    this.boundInstancesChangedListener = this.handleInstancesChange.bind( this );

    // @private {function} - Handles changes to focus, adding or removing listeners
    this.boundFocusListener = this.handleFocusChange.bind( this );
    this.focusProperty.link( this.boundFocusListener );

  }

  /**
   * When Focus changes, remove any listeners that were attached from last Focus and
   * add new listeners if focus has a new value.
   * @private
   *
   * @param {Focus} focus
   */
  handleFocusChange( focus ) {
    this.removeDisplayedListeners();

    if ( focus ) {
      this.addDisplayedListeners( focus );
    }
  }

  /**
   * When the Trail becomes invisible, Focus should be set to null.
   * @private
   */
  handleTrailVisibilityChange() {
    if ( this.visibilityTracker && !!this.visibilityTracker.trailVisibleProperty.value ) {
      this.focusProperty.value = null;
      this.onRemoveFocus();
    }
  }

  /**
   * If there are no more Instances for the Node with focus it has been removed from
   * the scene graph and so Focus should be set to null.
   * @private
   *
   * @param {Instance} instance
   */
  handleInstancesChange( instance ) {
    if ( instance.node.instances.length === 0 ) {
      this.focusProperty.value = null;
      this.onRemoveFocus();
    }
  }

  /**
   * Add listeners that watch when the Displayed state of the Node with Focus has changed,
   * including visibility of the trail and attachment to a scene graph.
   * @private
   *
   * @param {Focus} focus
   */
  addDisplayedListeners( focus ) {
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
   * @private
   */
  removeDisplayedListeners() {
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

  /**
   * @public
   */
  dispose() {

    // this disposes the TrailVisibilityTracker and removes any listeners on the Node
    this.removeDisplayedListeners();
    this.focusProperty.unlink( this.boundFocusListener );

    this.node = null;
    this.visibilityTracker = null;
    this.focusProperty = null;
  }
}

scenery.register( 'FocusDisplayedController', FocusDisplayedController );
export default FocusDisplayedController;