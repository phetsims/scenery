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

define( function( require ) {
  'use strict';

  var BooleanProperty = require( 'AXON/BooleanProperty' );
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  /**
   * @public
   * @constructor
   * @extends {BooleanProperty}
   *
   * @param {Node} node
   * @param {Display} display
   * @param {Object} [options] - Passed through to the BooleanProperty
   */
  function DisplayedProperty( node, display, options ) {
    BooleanProperty.call( this, false, options );

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
    var instances = node.instances;
    for ( var i = 0; i < instances.length; i++ ) {
      this.addedInstance( instances[ i ] );
    }
  }

  scenery.register( 'DisplayedProperty', DisplayedProperty );

  inherit( BooleanProperty, DisplayedProperty, {

    /**
     * Checks whether the node was displayed and updates the value of this Property.
     * @private
     */
    updateValue: function() {
      this.value = this.node.wasDisplayed( this.display );
    },

    /**
     * Adds a listener to one of the node's instances.
     * @private
     *
     * @param {Instance} instance
     */
    addedInstance: function( instance ) {
      instance.onStatic( 'visibility', this.updateListener );
      this.updateValue();
    },

    /**
     * Removes a listener from one of the node's instances.
     * @private
     *
     * @param {Instance} instance
     */
    removedInstance: function( instance ) {
      instance.offStatic( 'visibility', this.updateListener );
      this.updateValue();
    },

    /**
     * Releases references to avoid memory leaks.
     * @public
     * @override
     */
    dispose: function() {
      // Remove any instances the node may still have
      var instances = this.node.instances;
      for ( var i = 0; i < instances.length; i++ ) {
        this.removedInstance( instances[ i ] );
      }

      this.node.offStatic( 'addedInstance', this.addedInstancelistener );
      this.node.offStatic( 'removedInstance', this.removedInstancelistener );

      BooleanProperty.prototype.dispose.call( this );
    }
  } );

  return DisplayedProperty;
} );
