//  Copyright 2002-2014, University of Colorado Boulder

/**
 * The FocusLayer contains any focus highlights that are shown *outside* of particular nodes, such as
 * a focus rectangle, or graphical cursors.  Examples of highlights shown *within* nodes could be highlighting changes
 * or animation.
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var Node = require( 'SCENERY/nodes/Node' );
  var Input = require( 'SCENERY/input/Input' );
  var FocusRectangle = require( 'SCENERY/accessibility/FocusRectangle' );
  var FocusCursor = require( 'SCENERY/accessibility/FocusCursor' );
  var DerivedProperty = require( 'AXON/DerivedProperty' );
  var Property = require( 'AXON/Property' );
  var Events = require( 'AXON/Events' );

  /**
   * @constructor
   */
  function FocusLayer() {

    var expand = 5;

    //Dummy property to get DerivedProperty to play nice.
    // TODO: We either need a better API for this or a new pattern (could be event.trigger)
    var transformProperty = new Property( 0 );
    var events = new Events();

    var focusedBoundsProperty = new DerivedProperty( [ Input.focusedInstanceProperty, transformProperty ], function( focusedInstance, transform ) {
      if ( focusedInstance ) {
        var b = focusedInstance.node.getGlobalBounds();

        // TODO: A real live dot.Rectangle
        // TODO: Move expand to component classes
        return {
          x:      b.left - expand,
          y:      b.top - expand,
          width:  b.width + expand * 2,
          height: b.height + expand * 2
        };
      }
      else {
        return null;
      }
    } );

    var firstOne = true;
    var transformListener = function() {
      //transformProperty.value = transformProperty.value + 1;
      if ( firstOne ) {
        firstOne = false;
      }
      else {

        var b = Input.focusedInstanceProperty.value.node.getGlobalBounds();

        // TODO: A real live dot.Rectangle
        // TODO: Move expand to component classes
        var c = {
          x:      b.left - expand,
          y:      b.top - expand,
          width:  b.width + expand * 2,
          height: b.height + expand * 2
        };


        events.trigger( 'transformChanged', c );
        console.log( 'transformchanged' );
      }
    };


    Input.focusedInstanceProperty.link( function( focusedInstance ) {
      if ( focusedInstance ) {
        focusedInstance.relativeTransform.addListener( transformListener ); // when our relative transform changes, notify us in the pre-repaint phase
        console.log( 'addprecompute' );
        focusedInstance.relativeTransform.addPrecompute(); // trigger precomputation of the relative transform, since we will always need it when it is updated
        console.log( '/addprecompute' );
        firstOne = true;
      }
    } );

    var focusIndicatorProperty = new DerivedProperty( [ Input.focusedInstanceProperty ], function( focusedInstance ) {
      if ( focusedInstance ) {
        return focusedInstance.node.focusIndicator || 'rectangle';
      }
      else {
        return null;
      }
    } );

    this.focusRectangle = new FocusRectangle( focusedBoundsProperty, focusIndicatorProperty, events );
    this.focusCursor = new FocusCursor( focusedBoundsProperty, focusIndicatorProperty, events );

    Node.call( this, { children: [ this.focusRectangle, this.focusCursor ] } );
  }

  return inherit( Node, FocusLayer );
} );