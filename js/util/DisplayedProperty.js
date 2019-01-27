// Copyright 2018, University of Colorado Boulder

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

define( require => {
  'use strict';

  const BooleanProperty = require( 'AXON/BooleanProperty' );
  const scenery = require( 'SCENERY/scenery' );

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
      this.addedInstancelistener = this.addedInstance.bind( this );
      this.removedInstancelistener = this.removedInstance.bind( this );

      node.onStatic( 'addedInstance', this.addedInstancelistener );
      node.onStatic( 'removedInstance', this.removedInstancelistener );

      // Add any instances the node may already have/
      const instances = node.instances;
      for ( let i = 0; i < instances.length; i++ ) {
        this.addedInstance( instances[ i ] );
      }
    }

    /**
     * Checks whether the node was displayed and updates the value of this Property.
     * @private
     */
    updateValue() {
      this.value = this.node.wasDisplayed( this.display );
    }

    /**
     * Adds a listener to one of the node's instances.
     * @private
     *
     * @param {Instance} instance
     */
    addedInstance( instance ) {
      instance.onStatic( 'visibility', this.updateListener );
      this.updateValue();
    }

    /**
     * Removes a listener from one of the node's instances.
     * @private
     *
     * @param {Instance} instance
     */
    removedInstance( instance ) {
      instance.offStatic( 'visibility', this.updateListener );
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
        this.removedInstance( instances[ i ] );
      }

      this.node.offStatic( 'addedInstance', this.addedInstancelistener );
      this.node.offStatic( 'removedInstance', this.removedInstancelistener );

      super.dispose();
    }
  }

  return scenery.register( 'DisplayedProperty', DisplayedProperty );
} );
