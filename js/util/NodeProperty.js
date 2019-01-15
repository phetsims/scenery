// Copyright 2018, University of Colorado Boulder

/**
 * An axon Property for a Node.  When the specified trigger event (like 'pickability') occurs, the Property value
 * changes according to the getter (like 'pickable'), which may have any type.  This relies on guards in both Property
 * and Node to prevent cycles--if we detect an infinite loop for a case, we will need to add an epsilon tolerance
 * in the corresponding Node setter to short circuit the lop.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Jonathan Olson (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var Property = require( 'AXON/Property' );
  var scenery = require( 'SCENERY/scenery' );

  /**
   * @param {Node} node
   * @param {string} trigger - the Node trigger that will cause this NodeProperty to update, such as 'bounds', or 'opacity'
   * @param {string} attribute - for example 'left' or 'centerBottom' or 'visible' or 'pickable'
   * @param {Object} [options]
   */
  function NodeProperty( node, trigger, attribute, options ) {
    assert && assert( typeof attribute === 'string', 'wrong type for getLocation' );
    var self = this;

    // @private {Node}
    this.node = node;

    // @private {string} - for disposal
    this.trigger = trigger;

    // @private {string}
    this.attribute = attribute;

    // Read-only Property that describes a part relative to the bounds of the node.
    Property.call( this, node[ attribute ], options );

    // @private {function} - When the node event is triggered, get the new value and set it to this Property
    this.changeListener = function() {
      self.set( node[ attribute ] );
    };

    // onStatic (as opposed to 'on') avoids array allocation, but means the listener cannot cause disposal of this node.
    node.onStatic( trigger, this.changeListener );
  }

  scenery.register( 'NodeProperty', NodeProperty );

  return inherit( Property, NodeProperty, {

    /**
     * Unlinks listeners when disposed.  Must be called before the corresponding Node is disposed.
     * @public
     */
    dispose: function() {
      assert && assert( !this.node.isDisposed, 'NodeProperty should be disposed before corresponding Node is disposed' );
      this.node.offStatic( this.trigger, this.changeListener );
      Property.prototype.dispose.call( this );
      this.node = null;
    },

    /**
     * Updates the value of this node, overridden to update the Node value
     * @override
     * @param {*} value - the new value this Property will take, which is different than the previous value.
     * @protected - can be overridden.
     */
    setValueAndNotifyListeners: function( value ) {

      // Set the node value first, as if it was the first link listener.
      this.node[ this.attribute ] = value;

      // Notify the other listeners
      Property.prototype.setValueAndNotifyListeners.call( this, value );
    }
  } );
} );