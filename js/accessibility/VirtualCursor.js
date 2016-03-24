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
  var DATA_VISITED_LINEAR = 'data-visited-linear';

  function VirtualCursor() {

    /**
     * A timed queue that can be used to queue descriptions to the text output.  Usually used for
     * 'aria-live' updates.  If multiple updates are fired at once, they are added to the queue
     * in the order that they occur.
     * 
     * @param {number} defaultDelay - default delay for items in the queue
     */
    var TimedQueue = function( defaultDelay ) {
      Object.call( this );

      // @private - track when we are running so we do not try to run the queue while it is already running
      this.running = false;

      // @private - the queue of items to be read, populated with objects like
      // { callback: {function}, delay: {number} };
      this.queue = [];

      // @private - current index in the queue
      this.index = 0;

      // @private - default delay of five seconds
      this.defaultDelay = defaultDelay || 5000;
    };

    inherit( Object, TimedQueue, {

      /**
       * Add a callback to this queue, with a delay
       * 
       * @param {function} callBack - callback fired by this addition
       * @param {number} [delay]- optional delay for this item
       */
      add: function( callBack, delay ) {
        var thisQueue = this;
        this.queue.push( {
          callBack: callBack,
          delay: delay || thisQueue.defaultDelay
        } );
      },

      /**
       * Run through items in the queue, starting at index
       * @param  {number} index
       */
      run: function( index ) {
        if ( !this.running ) {
          this.index = index || 0;
          this.next();
        }
      },

      /**
       * Remove all items from the queue
       */
      clear: function() {
        this.queue = [];
      },

      /**
       * Get the next item in the queue, then delaying after firing callback
       */
      next: function() {

        this.running = true;
        var thisQueue = this;
        var i = this.index++;

        var active = this.queue[i];
        var next = this.queue[ this.index ];

        // return and set running flag to false if there are no items in the queue
        var endRun = function() {
          thisQueue.running = false;
          thisQueue.clear();
        };

        if( !active ) {
          endRun();
          return;
        }

        // fire the callback function
        active.callBack();

        if( next ) {
          setTimeout( function() {
            thisQueue.next();
          }, active.delay || thisQueue.defaultDelay );
        }
        else {
          endRun();
          return;
        }
      }
    } );

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
      if ( element.hidden ) {
        return null;
      }

      // search for elements that will have content that should be shown
      if ( element.tagName === 'P' ) {
        return element.textContent;
      }
      if( element.tagName === 'H1' ) {
        return 'heading level 1 ' + element.textContent;
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

      return null;
    };

    var clearVisited = function( element, visitedFlag ) {
      element.removeAttribute( visitedFlag );
      for ( var i = 0; i < element.children.length; i++ ) {
        clearVisited( element.children[ i ], visitedFlag );
      }
    };

    /**
     * Go to next element in the parallel DOM that has accessible text content.  If an element has
     * text content, it is returned.  Otherwise,
     * @param {DOMElement} element
     * @return {string} visitedFlag - a flag for which 'data-*' attribute to set as we search through the tree
     */
    var goToNextItem = function( element, visitedFlag ) {
      if ( getAccessibleText( element ) ) {
        if ( !element.getAttribute( visitedFlag ) ) {
          element.setAttribute( visitedFlag, true );
          return element;
        }
      }
      for ( var i = 0; i < element.children.length; i++ ) {
        var nextElement = goToNextItem( element.children[ i ], visitedFlag );

        if ( nextElement ) {
          return nextElement;
        }
      }
    };

    /**
     * Get all 'element' nodes of the parent element, placing them in an array for easy traversal
     * Note that this includes all elements, even those that are 'hidden'.
     *
     * TODO: Can this replace getLinearDOM and goToNextItem?
     * 
     * @param  {DOMElement} element - parent element for which you want an array of all children
     * @return {array<DOMElement>}
     */
    var getLinearDOMElements = function( element ) {
      // gets all descendent children for the element
      var children = element.getElementsByTagName( '*' );
      var linearDOM = [];
      for( var i = 0; i < children.length; i++ ) {
        if( children[i].nodeType === Node.ELEMENT_NODE ) {
          linearDOM[i] = ( children[ i ] );
        }
      }
      return linearDOM;
    };

    /**
     * Get a 'linear' representation of the DOM, collapsing the accessibility tree into an array that
     * can be traversed.
     *
     * @param {DOMElement} element
     * @return {array<DOMElement>} linearDOM
     */
    var getLinearDOM = function( element ) {

      clearVisited( element, DATA_VISITED_LINEAR );

      var linearDOM = [];

      var nextElement = goToNextItem( element, DATA_VISITED_LINEAR );
      while ( nextElement ) {
        linearDOM.push( nextElement );
        nextElement = goToNextItem( element, DATA_VISITED_LINEAR );
      }

      return linearDOM;
    };

    // if the user presses right or down, move the cursor to the next dom element with accessibility markup
    // It will be difficult to synchronize the virtual cursor with tab navigation.  For now, tab/shift+tab
    // will take us to the next/previos description.
    document.addEventListener( 'keydown', function( k ) {
      var selectedElement = null;
      var accessibilityDOMElement = document.body.getElementsByClassName( 'accessibility' )[ 0 ];

      // forward traversal
      if ( k.keyCode === 39 || k.keyCode === 40 || ( k.keyCode === 9 && !k.shiftKey ) ) {

        // we do not want to go to the next forms element if tab (yet)
        k.preventDefault();

        selectedElement = goToNextItem( accessibilityDOMElement, DATA_VISITED );

        if ( !selectedElement ) {
          clearVisited( accessibilityDOMElement, DATA_VISITED );
          selectedElement = goToNextItem( accessibilityDOMElement, DATA_VISITED );
        }
      }

      // backwards traversal on up/left/shift+tab
      else if ( k.keyCode === 37 || k.keyCode === 38 || ( k.shiftKey && k.keyCode === 9 ) ) {

        // we do not want to go to the next forms element if tab (yet)
        k.preventDefault();

        var listOfAccessibleElements = getLinearDOM( accessibilityDOMElement );

        var foundAccessibleText = false;
        for ( var i = listOfAccessibleElements.length - 1; i >= 0; i-- ) {
          if ( listOfAccessibleElements[ i ].getAttribute( DATA_VISITED ) ) {
            listOfAccessibleElements[ i ].removeAttribute( DATA_VISITED );

            if ( i !== 0 ) {
              selectedElement = listOfAccessibleElements[ i - 1 ];
              foundAccessibleText = true;
            }
            else {
              selectedElement = null;
              foundAccessibleText = false;
            }

            break;
          }
        }

        // Wrap backwards, going from the end of the linearized DOM
        if ( !foundAccessibleText ) {
          selectedElement = listOfAccessibleElements[ listOfAccessibleElements.length - 1 ];

          for ( i = 0; i < listOfAccessibleElements.length - 1; i++ ) {
            listOfAccessibleElements[ i ].setAttribute( DATA_VISITED, true );
          }
        }
      }

      if ( selectedElement ) {
        if ( selectedElement.getAttribute( 'tabindex' ) || selectedElement.tagName === 'BUTTON' ) {
          selectedElement.focus();
        }

        // print the output to the iFrame
        var accessibleText = getAccessibleText( selectedElement );
        parent && parent.updateAccessibilityReadoutText && parent.updateAccessibilityReadoutText( accessibleText );
      }
    } );

    // create a queue for aria live textcontent changes
    // changes to elements with aria-live text content will be added to queue by element mutation observers
    var ariaLiveQueue = new TimedQueue( 5000 );

    var updateLiveElementList = function() {

      // remove all observers from live elements to prevent a memory leak
      if( window.liveElementList ) {
        for( var key in window.liveElementList ) {
          if( window.liveElementList.hasOwnProperty( key ) ) {
            window.liveElementList[key].observer.disconnect();
          }
        }
      }

      // clear the list
      window.liveElementList = {};

      // get a linear representation of the DOM
      var accessibilityDOMElement = document.body.getElementsByClassName( 'accessibility' )[ 0 ];
      var linearDOM = getLinearDOMElements( accessibilityDOMElement );

      // search the DOM for elements with 'aria-live' attributes
      for( var i = 0; i < linearDOM.length; i++ ) {
        var domElement = linearDOM[ i ];
        if( domElement.getAttribute( 'aria-live' ) ) {

          // create an observer for this element
          var observer = new MutationObserver( function( mutations ) {
            mutations.forEach( function( mutation ) {
              var updatedText = mutation.addedNodes[0].data;
              ariaLiveQueue.add( function() {
                console.log( updatedText );
                parent &&
                  parent.updateAccessibilityReadoutText &&
                    parent.updateAccessibilityReadoutText( updatedText );

              }, 2000 );
            } );
          } );

          window.liveElementList[ domElement.id.toString() ] = {
            'domElement': domElement,
            'observer': observer 
          };

          // listen for changes to the subtree in case children of the aria-live parent change their textContent
          var observerConfig = { childList: true, characterData: true, subtree: true };

          observer.observe( domElement, observerConfig );
        }        
      }
    };

    // look for new live elements and add listeners to them every five seconds
    // TODO: Is there a better way to search through the DOM for new aria-live elements?
    var searchForLiveElements = function() {
      updateLiveElementList();
      setTimeout( searchForLiveElements, 5000 );
    };
    searchForLiveElements();

    // run through the active queue of aria live elements
    var step = function() {
      ariaLiveQueue.run();
      setTimeout( step, 100 );
    };
    step();
  }

  return inherit( Object, VirtualCursor, {} );

} );