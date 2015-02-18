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

  var trailToGlobalBounds = function( trail ) {
    return trail.parentToGlobalBounds( trail.lastNode().bounds );
  };

  var cleanup = [];

  /**
   * @param {Object} [options] - optional configuration, see constructor
   * @constructor
   */
  function FocusLayer( options ) {

    options = _.extend( {

      /**
       * tweenFactory - optional tween factory that will be used to update the location of the focus region. This object
       * must conform to the API as used here (somewhat complex). If not provided, the default (instant) tween factory
       * will be used. To show animated focus regions, pass in an instance of sole/TWEEN (as done in JOIST/Sim.js)
       */
      tweenFactory: FocusLayer.INSTANT_TWEEN_FACTORY
    }, options );

    // Return an object optimal for TWEEN, containing only the required attributes for animation
    // This is important because TWEEN.js calls all fields + getters to determine initial state
    // So we must create a minimal pruned object of only the values we wish to animate.
    var boundsToObject = function( bounds ) {
      return { x: bounds.x, y: bounds.y, width: bounds.width, height: bounds.height };
    };

    var tween = null;

    // Animates when focused instance changes.  Jumps (discrete) when target object transform changes.
    var focusedBoundsProperty = new Property();
    Input.focusedTrailProperty.link( function( focusedTrail, previousFocusedTrail ) {
      if ( focusedTrail && previousFocusedTrail ) {

        var focusRectangle = trailToGlobalBounds( focusedTrail );
        var previousFocusRectangle;

        // Use the bounds of the previous node for starting animation point.
        // However, that node may have been removed from the scene graph.
        if ( previousFocusedTrail ) {
          previousFocusRectangle = trailToGlobalBounds( previousFocusedTrail );
        }
        else {
          // TODO: Could replace this with storing the previous bounds from the last callback
          previousFocusRectangle = trailToGlobalBounds( focusedTrail );
        }

        if ( tween ) {
          tween.stop();
          tween = null;
        }

        // For accessibility animation, scenery requires the TWEEN.js library
        // If this API usage is changed, the INSTANT_TWEEN_FACTORY must also be changed correspondingly.
        tween = new options.tweenFactory.Tween( boundsToObject( previousFocusRectangle ) ).
          to( boundsToObject( focusRectangle ), 300 ).
          easing( options.tweenFactory.Easing.Cubic.InOut ).
          onUpdate( function() {
            focusedBoundsProperty.set( { x: this.x, y: this.y, width: this.width, height: this.height } );
          } ).
          onComplete( function() {
            tween = null;
          } ).
          start();
      }
      else if ( focusedTrail && previousFocusedTrail === null ) {
        focusedBoundsProperty.value = trailToGlobalBounds( focusedTrail );
      }
      else {
        focusedBoundsProperty.value = null;
      }

      // Detach listeners from the previous trail
      for ( var i = 0; i < cleanup.length; i++ ) {
        cleanup[ i ]();
      }
      cleanup.length = 0;

      // Attach listeners up the tree of the focused node so that when the bounds change we can update the focus rectangle
      if ( focusedTrail ) {

        // A function that will update the focus bounds based on the focusedTrail
        var updateFocusBounds = function() {

          // If the node was still animating, cancel the animation or it would animate to the wrong place.
          if ( tween ) {
            tween.stop();
            tween = null;
          }
          focusedBoundsProperty.value = trailToGlobalBounds( focusedTrail );
        };

        // For each node in the focused trail, add a listener for transform changes.
        focusedTrail.nodes.forEach( function( node ) {
          node.on( 'transform', updateFocusBounds );

          cleanup.push( function() {
            node.off( 'transform', updateFocusBounds );
          } );
        } );

        // When the node's bounds change, update the focus rectangle
        var lastNode = focusedTrail.lastNode();
        lastNode.on( 'bounds', updateFocusBounds );
        cleanup.push( function() {
          lastNode.off( 'bounds', updateFocusBounds );
        } );

        // TODO: When the node's visibility changes, we need a new focus trail.
      }
    } );


    // This property indicates which kind of focus region is being shown.  For instance, 'cursor' or 'rectangle'
    // TODO: Make it possible to add new focus types here on a simulation-by-simulation basis
    var focusIndicatorProperty = new DerivedProperty( [ Input.focusedTrailProperty ], function( focusedTrail ) {

      // the check for node existence seems necessary for handling appearing/disappearing popups
      if ( focusedTrail ) {
        return focusedTrail.lastNode().focusIndicator || 'rectangle';
      }
      else {
        return null;
      }
    } );

    this.focusRectangle = new FocusRectangle( focusedBoundsProperty, focusIndicatorProperty );
    this.focusCursor = new FocusCursor( focusedBoundsProperty, focusIndicatorProperty );

    Node.call( this, { children: [ this.focusRectangle, this.focusCursor ] } );
  }

  return inherit( Node, FocusLayer, {}, {

    // An implementation of the tween factory interface that shows instantly-moving focus regions without TWEEN.js support
    INSTANT_TWEEN_FACTORY: {
      Easing: { Cubic: { InOut: true } },
      Tween: function() {
        return {
          to: function( finalState ) {
            this.finalState = finalState;
            return this;
          },
          easing: function() {return this;},
          onUpdate: function( callback ) {
            this.callback = callback;
            return this;
          },
          onComplete: function() {return this;},
          start: function() {
            this.callback.call( this.finalState );
          },
          stop: function() {}
        };
      }
    }
  } );
} );
