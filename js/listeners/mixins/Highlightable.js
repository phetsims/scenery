// Copyright 2013-2017, University of Colorado Boulder

/**
 * [[EXPERIMENTAL]] Mixin for PressListener that adds highlight tracking (pressed or hovering)
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var scenery = require( 'SCENERY/scenery' );
  var Property = require( 'AXON/Property' );

  function HighlightableConstructor( options ) {
    var self = this;

    options = _.extend( {
      // TODO: add alternate highlight schemes?
      isHighlightedProperty: new Property( false ) // allowing this to be overridden helps with button models
    }, options );

    this.isHighlightedProperty = options.isHighlightedProperty;

    Property.multilink( [ this.isOverProperty, this.isPressedProperty ], function( isOver, isPressed ) {
      self.isHighlightedProperty.value = isOver || isPressed;
    } );
  }

  var Highlightable = {
    mixin: function Highlightable( type ) {
      var proto = type.prototype;

      assert && assert( proto.mixesHoverable );

      proto.mixesHighlightable = true;

      return HighlightableConstructor;
    }
  };

  scenery.register( 'Highlightable', Highlightable );

  return Highlightable;
} );
