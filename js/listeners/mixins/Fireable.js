// Copyright 2013-2017, University of Colorado Boulder

/**
 * [[EXPERIMENTAL]] Mixin for PressListener that adds fire() with fireOnDown capability
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var scenery = require( 'SCENERY/scenery' );
  var protoAfter = require( 'SCENERY/listeners/mixins/protoAfter' );

  function FireableConstructor( options ) {
    options = _.extend( {
      fireOnDown: false,
      fire: null
    }, options );

    this._fireOnDown = options.fireOnDown;
    this._fireCallback = options.fire;
  }

  var Fireable = {
    mixin: function Fireable( type ) {
      var proto = type.prototype;

      assert && assert( proto.mixesHoverable );

      proto.mixesFireable = true;

      proto.fire = function( event ) {
        this._fireCallback && this._fireCallback( event );
      };

      protoAfter( proto, 'press', function press( event ) {
        if ( this._fireOnDown ) {
          this.fire( event );
        }
      } );

      protoAfter( proto, 'release', function release() {
        if ( !this._fireOnDown && this.isOverProperty.value && !this.wasInterrupted ) {
          this.fire();
        }
      } );

      return FireableConstructor;
    }
  };

  scenery.register( 'Fireable', Fireable );

  return Fireable;
} );
