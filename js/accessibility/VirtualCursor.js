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

  var DATA_VISITED = 'data-visited';

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

      // filter out structural elements that do not have accessible text
      if ( element.getAttribute( 'class' ) === 'ScreenView' ) {
        return null;
      }
      if ( element.getAttribute( 'aria-hidden' ) ) {
        return null;
      }

      // search for accessibility mark up in the pararllel DOM, these elements have accessible text
      if ( element.getAttribute( 'aria-labelledby' ) ) {
        var labelElement = document.getElementById( element.getAttribute( 'aria-labelledby' ) );
        if ( !labelElement ) {
          console.log( 'Missing labelled element with aria-labelledby id' );
          return null;
        }
        return labelElement.textContent;
      }
      if ( element.getAttribute( 'aria-describedby' ) ) {
        var descriptionElement = document.getElementById( element.getAttribute( 'aria-describedby' ) );
        if ( !descriptionElement ) {
          console.log( 'Missing labelled element with aria-describedby id' );
          return null;
        }
        return descriptionElement.textContent;
      }
      if ( element.tagName === 'P' ) {
        return element.textContent;
      }
      if ( element.tagName === 'H2' ) {
        return 'heading level 2 ' + element.textContent;
      }
      if ( element.tagName === 'H3' ) {
        return 'heading level 3 ' + element.textContent;
      }
      if ( element.tagName === 'BUTTON' ) {
        return element.textContent + ' button';
      }
      if ( element.tagName === 'INPUT' ) {
        if ( element.type === 'reset' ) {
          return element.getAttribute( 'value' );
        }
        if ( element.type === 'checkbox' ) {
          var checkedString = element.checked ? ' checked' : ' not checked';
          return element.textContent + ' checkbox' + checkedString;
        }
      }


      // search for elements in the parallel DOM that will have implicit accessible text without markup

      return null;
    };

    var clearVisited = function( element ) {
      element.removeAttribute( DATA_VISITED );
      for ( var i = 0; i < element.children.length; i++ ) {
        clearVisited( element.children[ i ] );
      }
    };

    /**
     * Go to next element in the parallel DOM that has accessible text content.  If an element has
     * text content, it is returned.  Otherwise,
     * @param element [description]
     * @return {[type]}         [description]
     */
    var goToNextItem = function( element ) {
      if ( getAccessibleText( element ) ) {


        if ( !element.getAttribute( DATA_VISITED ) ) {
          element.setAttribute( DATA_VISITED, true );
          return element;
        }
        // else if ( element === selectedElement ) {

        //   // Running the first pass depth-first search from the root has found the previously selected item
        //   // so now we can continue the search and return the next focusable item.
        //   element.setAttribute( 'data-visited', true );
        // }
      }
      for ( var i = 0; i < element.children.length; i++ ) {
        var nextElement = goToNextItem( element.children[ i ] );

        if ( nextElement ) {
          return nextElement;
        }
      }
    };

    // if the user presses right or down, move the cursor to the next dom element with accessibility markup
    // It will be difficult to synchronize the virtual cursor with tab navigation so we are not implementing
    // this for now.
    document.addEventListener( 'keydown', function( k ) {
      if ( k.keyCode === 39 || k.keyCode === 40 ) {

        // TODO: access this once?
        //debugger;
        var accessibilityDOMElement = document.body.getElementsByClassName( 'accessibility' )[ 0 ];
        selectedElement = goToNextItem( accessibilityDOMElement );

        if ( !selectedElement ) {
          clearVisited( accessibilityDOMElement );
          selectedElement = goToNextItem( accessibilityDOMElement );
        }
        var accessibleText = getAccessibleText( selectedElement );
        parent && parent.updateAccessibilityReadoutText && parent.updateAccessibilityReadoutText( accessibleText );
        //console.log( accessibleText );
      }
    } );

  }

  return inherit( Object, VirtualCursor, {} );
} );