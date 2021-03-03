// Copyright 2018-2020, University of Colorado Boulder

/**
 * A property that is true when the node appears on the given display.
 *
 * Note that a node can appear on a display even after it has been removed from the scene graph, if
 * Display.updateDisplay has not yet been called since it was removed. So generally this Property will only update
 * as a result of Display.updateDisplay() being called.
 *
 * Be careful to dispose of these, since it WILL result in a permanent memory leak otherwise (Instance objects are
 * pooled, and if the listener is not removed, it will stay around forever).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import BooleanProperty from '../../../axon/js/BooleanProperty.js';
import scenery from '../scenery.js';

class DisplayedProperty extends BooleanProperty {
  /**
   * @public
   * @extends {BooleanProperty}
   *
   * @param {Node} node
   * @param {Display} display
   * @param {Object} [options] - Passed through to the BooleanProperty
   */
  constructor( node, display, options ) {
    super( false, options );

    // @private {Node}
    this.node = node;

    // @private {Display}
    this.display = display;

    // @private {function}
    this.updateListener = this.updateValue.bind( this );
    this.changedInstanceListener = this.changedInstance.bind( this );

    node.changedInstanceEmitter.addListener( this.changedInstanceListener );

    // Add any instances the node may already have/
    const instances = node.instances;
    for ( let i = 0; i < instances.length; i++ ) {
      this.changedInstance( instances[ i ], true );
    }
  }

  /**
   * Checks whether the node was displayed and updates the value of this Property.
   * @private
   */
  updateValue() {
    this.value = this.node.wasVisuallyDisplayed( this.display );
  }

  /**
   * Called when an instance is changed or added (based on the boolean flag).
   * @private
   *
   * @param {Instance} instance
   * @param {boolean} added
   */
  changedInstance( instance, added ) {
    if ( added ) {
      instance.visibleEmitter.addListener( this.updateListener );
    }
    else {
      instance.visibleEmitter.removeListener( this.updateListener );
    }

    this.updateValue();
  }

  /**
   * Releases references to avoid memory leaks.
   * @public
   * @override
   */
  dispose() {
    // Remove any instances the node may still have
    const instances = this.node.instances;
    for ( let i = 0; i < instances.length; i++ ) {
      this.changedInstance( instances[ i ], false );
    }

    this.node.changedInstanceEmitter.removeListener( this.changedInstanceListener );

    super.dispose();
  }
}

scenery.register( 'DisplayedProperty', DisplayedProperty );
export default DisplayedProperty;