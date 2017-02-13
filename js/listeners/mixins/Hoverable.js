// Copyright 2013-2016, University of Colorado Boulder

/**
 * [[EXPERIMENTAL]] Mixin for PressListener that adds tracking for how many pointers are over the listener.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var scenery = require( 'SCENERY/scenery' );
  var Property = require( 'AXON/Property' );
  var protoAfter = require( 'SCENERY/listeners/mixins/protoAfter' );

  function HoverableConstructor( options ) {
    var self = this;

    options = _.extend( {
      isOverProperty: new Property( false ) // allowing this to be overridden helps with button models
    }, options );

    this.overCountProperty = new Property( 0 );
    this.isOverProperty = options.isOverProperty;

    this.overCountProperty.link( function( overCount ) {
      self.isOverProperty.value = overCount > 0;
    } );
  }

  var Hoverable = {
    mixin: function Hoverable( type ) {
      var proto = type.prototype;

      proto.mixesHoverable = true;

      protoAfter( proto, 'enter', function enter( event ) {
        this.overCountProperty.value++;
      } );

      protoAfter( proto, 'exit', function exit( event ) {
        assert && assert( this.overCountProperty.value >= 0, 'Exit event not matched by an enter event' );

        this.overCountProperty.value--;
      } );

      return HoverableConstructor;
    }
  };

  scenery.register( 'Hoverable', Hoverable );

  return Hoverable;
} );
