// Copyright 2002-2014, University of Colorado Boulder

/**
 *
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var Rectangle = require( 'SCENERY/nodes/Rectangle' );
  var Property = require( 'AXON/Property' );

  /**
   *
   * @constructor
   */
  function FocusRectangle( focusedBoundsProperty, focusIndicatorProperty ) {
    var focusRectangle = this;

    Rectangle.call( this, 0, 0, 0, 0, 0, 0, {
      stroke: 'rgb(178,35,154)',
      visible: false,
      lineWidth: 3
    } );

    var expand = 2.5;
    focusedBoundsProperty.link( function( targetBounds ) {
      if ( targetBounds ) {
        focusRectangle.setRect( targetBounds.x - expand, targetBounds.y - expand, targetBounds.width + expand * 2, targetBounds.height + expand * 2, 6, 6 );
      }
    } );

    Property.multilink( [ focusedBoundsProperty, focusIndicatorProperty ], function( focusedBounds, focusIndicator ) {
      var visible = focusedBounds !== null && focusIndicator === 'rectangle';
      focusRectangle.visible = visible;
    } );

  }

  return inherit( Rectangle, FocusRectangle );
} );