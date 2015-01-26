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

  /**
   * @constructor
   */
  function FocusLayer() {

    // Return an object optimal for TWEEN
    var boundsToObject = function( bounds ) {
      return { x: bounds.x, y: bounds.y, width: bounds.width, height: bounds.height };
    };

    var tween = null;

    // Animates when focused instance changes.  Jumps (discrete) when target object transform changes.
    var focusedBoundsProperty = new Property();
    Input.focusedInstanceProperty.link( function( focusedInstance, previousFocusedInstance ) {
      if ( focusedInstance && previousFocusedInstance && focusedInstance.node ) {

        var focusRectangle = focusedInstance.node.getGlobalBounds();
        var previousFocusRectangle;

        // Use the bounds of the previous node for starting animation point.
        // However, that node may have been removed from the scene graph.
        if ( previousFocusedInstance.node ) {
          previousFocusRectangle = previousFocusedInstance.node.getGlobalBounds();
        }
        else {
          // TODO: Could replace this with storing the previous bounds from the last callback
          previousFocusRectangle = focusedInstance.node.getGlobalBounds();
        }

        if ( tween ) {
          tween.stop();
          tween = null;
        }
        // For accessibility animation, scenery requires the TWEEN.js library
        tween = new TWEEN.Tween( boundsToObject( previousFocusRectangle ) ).
          to( boundsToObject( focusRectangle ), 300 ).
          easing( TWEEN.Easing.Cubic.InOut ).
          onUpdate( function() {
            focusedBoundsProperty.set( { x: this.x, y: this.y, width: this.width, height: this.height } );
          } ).
          onComplete( function() {
            tween = null;
          } ).
          start();
      }
      else if ( focusedInstance && previousFocusedInstance === null ) {
        focusedBoundsProperty.value = focusedInstance.node.getGlobalBounds();
      }
      else {
        focusedBoundsProperty.value = null;
      }
    } );

    // There is a spurious transform listener callback when registering a listener (perhaps?)
    // TODO: This spurious event needs to be discussed and reviewed with Jon Olson to make sure
    // TODO: it is not a long term maintenance issue
    var firstOne = true;
    var transformListener = function() {
      //transformProperty.value = transformProperty.value + 1;
      if ( firstOne ) {
        firstOne = false;
      }
      else {
        if ( tween ) {
          tween.stop();
          tween = null;
        }
        focusedBoundsProperty.value = Input.focusedInstanceProperty.value.node.getGlobalBounds();
      }
    };

    Input.focusedInstanceProperty.link( function( focusedInstance, previousFocusedInstance ) {
      if ( previousFocusedInstance ) {
        previousFocusedInstance.relativeTransform.removeListener( transformListener );
        previousFocusedInstance.relativeTransform.removePrecompute();
      }
      if ( focusedInstance ) {
        focusedInstance.relativeTransform.addListener( transformListener ); // when our relative transform changes, notify us in the pre-repaint phase
        focusedInstance.relativeTransform.addPrecompute(); // trigger precomputation of the relative transform, since we will always need it when it is updated
        firstOne = true;

        // TODO: What if parent(s) transforms change?
      }
    } );

    var focusIndicatorProperty = new DerivedProperty( [ Input.focusedInstanceProperty ], function( focusedInstance ) {

      // the check for node existence seems necessary for handling appearing/disappearing popups
      if ( focusedInstance && focusedInstance.node ) {
        return focusedInstance.node.focusIndicator || 'rectangle';
      }
      else {
        return null;
      }
    } );

    this.focusRectangle = new FocusRectangle( focusedBoundsProperty, focusIndicatorProperty );
    this.focusCursor = new FocusCursor( focusedBoundsProperty, focusIndicatorProperty );

    Node.call( this, { children: [ this.focusRectangle, this.focusCursor ] } );
  }

  return inherit( Node, FocusLayer );
} );