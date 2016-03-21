// Copyright 2016, University of Colorado Boulder

/**
 *
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Jesse Greenberg
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );

  function VirtualCursor() {
    var selectedElement = null;

    /**
     * Get the accessible text for this element.  An element will have accessible text if it contains
     * accessible markup, of is one of many defined elements that implicitly have accessible text.
     *
     * @param  {DOMElement} element searching through
     * @return {string}
     */
    var getAccessibleText = function( element ) {

      console.log( 'checking', element );

      // filter out structural elements that do not have accessible text
      if ( element.getAttribute( 'class' ) === 'ScreenView' ) {
        return null;
      }

      // search for accessibility mark up in the pararllel DOM, these elements have accessible text
      if ( element.getAttribute( 'aria-labelledby' ) ) {
        var labelledElement = document.getElementById( element.getAttribute( 'aria-labelledby' ) );
        return labelledElement.textContent;
      }
      if ( element.tagName === 'P' ) {
        return element.textContent;
      }
      if ( element.tagName === 'H2' ) {
        return element.textContent;
      }

      // search for elements in the parallel DOM that will have implicit accessible text without markup

      return null;
    };

    /**
     * Go to next element in the parallel DOM that has accessible text content.  If an element has
     * text content, it is returned.  Otherwise,
     * @param  {[type]} element [description]
     * @return {[type]}         [description]
     */
    var goToNextItem = function( element, visited ) {
      for ( var i = 0; i < element.children.length; i++ ) {
        if ( getAccessibleText( element.children[ i ] ) ) {

          if ( visited ) {
            return element.children[ i ];
          }
          else if ( element.children[ i ] === selectedElement ) {

            // Running the first pass depth-first search from the root has found the previously selected item
            // so now we can continue the search and return the next focusable item.
            visited = true;
          }
        }
        else {
          var nextElement = goToNextItem( element.children[ i ], visited );

          if ( nextElement === null ) {
            continue;
          }
          else {
            return nextElement;
          }
        }
      }
    };

    // if the user pressed tab, right or down, move the cursor to the next dom element with accessibility markup
    document.addEventListener( 'keydown', function( k ) {
      if ( k.keyCode === 39 || k.keyCode === 40 || k.keyCode === 9 ) {
        console.log( 'moving forward' );

        // TODO: access this once?
        //debugger;
        var accessibilityDOMElement = document.body.getElementsByClassName( 'accessibility' )[ 0 ];
        var visited = selectedElement === null;
        selectedElement = goToNextItem( accessibilityDOMElement, visited );

        if ( !selectedElement ) {
          console.log( '--wrapped' );

          selectedElement = null;
          visited = selectedElement === null;
          selectedElement = goToNextItem( accessibilityDOMElement, visited );
        }
        console.log( getAccessibleText( selectedElement ) );
      }
    } );

  }

  return inherit( Object, VirtualCursor, {} );
} );