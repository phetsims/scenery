// Copyright 2018, University of Colorado Boulder

/**
 * An axon Property for the position relative to the bounds of a Node, which updates when the Node bounds update.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var Property = require( 'AXON/Property' );
  var scenery = require( 'SCENERY/scenery' );

  /**
   * @param {Node} node
   * @param {function|string} getLocation, for example:
   *                                       function(node){ return node.center.plusXY(node.width/4,5); }
   *                                       or 'leftBottom'
   * @param {Object} [options]
   */
  function NodePositionProperty( node, getLocation, options ) {
    assert && assert( typeof getLocation === 'string' || typeof getLocation === 'function', 'wrong type for getLocation' );
    options = _.extend( {

      // For convenience, additional offset relative to the given point
      dx: 0,
      dy: 0
    }, options );
    var self = this;

    // @private - for disposal
    this.node = node;

    // Gets the value
    var locationGetter = function() {
      var location = ( typeof getLocation === 'string' ) ? node[ getLocation ] : getLocation( node );
      location.x += options.dx;
      location.y += options.dy;
      assert && assert( !isNaN( location.x ), 'location.x must be a number' );
      assert && assert( !isNaN( location.y ), 'location.y must be a number' );
      return location;
    };

    // Read-only Property that describes a part relative to the bounds of the node.
    Property.call( this, locationGetter(), {
      isValidValue: function( value ) {
        return value.isVector2;// Cannot use instanceof since Axon cannot depend on Dot
      }
    } );

    // @private - When the node Bounds change, update the position property
    this.boundsChangeListener = function() {
      self.set( locationGetter() );
    };

    // onStatic (as opposed to 'on') avoids array allocation, but means the listener cannot cause disposal of this node.
    node.onStatic( 'bounds', this.boundsChangeListener );
  }

  scenery.register( 'NodePositionProperty', NodePositionProperty );

  return inherit( Property, NodePositionProperty, {

    /**
     * Unlinks listeners when disposed.
     * @public
     */
    dispose: function() {
      this.node.offStatic( 'bounds', this.boundsChangeListener );
      Property.prototype.dispose.call( this );
    }
  } );
} );