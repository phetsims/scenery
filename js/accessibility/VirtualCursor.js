// Copyright 2016, University of Colorado Boulder

/**
 *
 * @author Sam Reid (PhET Interactive Simulations)
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );

  function VirtualCursor() {
    var selectedElement = null;

    //var goToNextItem = function( element ) {
    //  element.children
    //};

    // if the user pressed tab, right or down, move the cursor to the next dom element with accessibility markup
    document.addEventListener( 'keydown', function( k ) {
      if ( k.keyCode === 39 || k.keyCode === 40 || k.keyCode === 9 ) {
        console.log( 'moving forward' );

        // TODO: access this once?
        //var accessibilityDOMElement = document.body.getElementsByClassName( 'accessibility' )[ 0 ];
        if ( selectedElement === null ) {
          //selectedElement = goToNextItem( accessibilityDOMElement );
        }
      }
    } );


    //left = 37
    //up = 38
    //right = 39
    //down = 40
  }

  return inherit( Object, VirtualCursor, {} );
} );