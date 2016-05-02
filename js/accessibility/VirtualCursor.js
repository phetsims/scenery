// Copyright 2016, University of Colorado Boulder

/**
 * Prototype virtual cursor.  The virtual cursor is typically used by asistive technologies to enable the user
 * to traverse content on the page. This is meant to prototype a tool that lets the team
 * and trusted translators 'view' the accessible content for a simulation.  The accessible content exsists in the 
 * accessible DOM, and the virtual cursor traverses this document.
 * 
 * The cursor moves with the 'tab' and 'arrow' keys.  'Tab', 'down arrow', and 'right arrow' move cursor to 
 * the next item, while 'shift + tab', 'left arrow', and 'up arrow' moves the cursor to the revious item.
 *
 * This prototype cursor assumes that the simulation is embedded in an iFrame, and the output is to be printed under
 * the iFrame, see balloons-and-static-electricity/screen-reader.html
 *
 * See https://github.com/phetsims/scenery/issues/538
 * 
 * @author Sam Reid (PhET Interactive Simulations)
 * @author Jesse Greenberg
 */
define( function( require ) {
  'use strict';

  // modules
  var inherit = require( 'PHET_CORE/inherit' );
  var scenery = require( 'SCENERY/scenery' );

  // constants
  // flags which are set on visited html elements during traversal
  var DATA_VISITED = 'data-visited';
  var DATA_VISITED_LINEAR = 'data-visited-linear';

  /**
   * Constructor.
   */
  function VirtualCursor() {

    var thisCursor = this;

    // handle traversal of the DOM with keyboard navigation
    document.addEventListener( 'keydown', function( k ) {
      var selectedElement = null;
      var accessibilityDOMElement = document.body.getElementsByClassName( 'accessibility' )[ 0 ];

      // forward traversal
      if ( k.keyCode === 39 || k.keyCode === 40 || ( k.keyCode === 9 && !k.shiftKey ) ) {

        // we do not want to go to the next forms element if tab (yet)
        k.preventDefault();

        selectedElement = thisCursor.goToNextItem( accessibilityDOMElement, DATA_VISITED );

        if ( !selectedElement ) {
          thisCursor.clearVisited( accessibilityDOMElement, DATA_VISITED );
          selectedElement = thisCursor.goToNextItem( accessibilityDOMElement, DATA_VISITED );
        }
      }

      // backwards traversal
      else if ( k.keyCode === 37 || k.keyCode === 38 || ( k.shiftKey && k.keyCode === 9 ) ) {

        // we do not want to go to the next forms element if tab (yet)
        k.preventDefault();

        var listOfAccessibleElements = thisCursor.getLinearDOM( accessibilityDOMElement );

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

      // if the element has a tabindex or is a button, navigation with the virtual cursor will set
      // it as the activeElement
      if ( selectedElement ) {
        if ( selectedElement.getAttribute( 'tabindex' ) || selectedElement.tagName === 'BUTTON' ) {
          selectedElement.focus();
        }

        // print the output to the iFrame
        var accessibleText = thisCursor.getAccessibleText( selectedElement );
        parent && parent.updateAccessibilityReadoutText && parent.updateAccessibilityReadoutText( accessibleText );
      }
    } );

    // create a queue for aria live textcontent changes
    // changes to elements with aria-live text content will be added to queue by mutation observers
    this.ariaLiveQueue = new TimedQueue( 5000 );

    // look for new live elements and add listeners to them every five seconds
    // TODO: Is there a better way to search through the DOM for new aria-live elements?
    var searchForLiveElements = function() {
      thisCursor.updateLiveElementList();
      setTimeout( searchForLiveElements, 5000 );
    };
    searchForLiveElements();

    // run through the active queue of aria live elements
    var step = function() {
      thisCursor.ariaLiveQueue.run();
      setTimeout( step, 100 );
    };
    step();
  }

  scenery.register( 'VirtualCursor', VirtualCursor );

  inherit( Object, VirtualCursor, {

    /**
     * Get the accessible text for this element.  An element will have accessible text if it contains
     * accessible markup, of is one of many defined elements that implicitly have accessible text.
     *
     * @param  {DOMElement} element searching through
     * @return {string}
     * @private
     */
    getAccessibleText: function( element ) {

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
        return 'Heading Level 1 ' + element.textContent;
      }
      if ( element.tagName === 'H2' ) {
        return 'Heading Level 2 ' + element.textContent;
      }
      if ( element.tagName === 'H3' ) {
        return 'Heading Level 3 ' + element.textContent;
      }
      if ( element.tagName === 'BUTTON' ) {
        return element.textContent + ' Button';
      }
      if ( element.tagName === 'INPUT' ) {
        if ( element.type === 'reset' ) {
          return element.getAttribute( 'value' ) + ' Button';
        }
        if ( element.type === 'checkbox' ) {
          var checkedString = element.checked ? ' Checked' : ' Not Checked';
          return element.textContent + ' Checkbox' + checkedString;
        }
      }

      return null;
    },

    /**
     * Get all 'element' nodes of the parent element, placing them in an array for easy traversal
     * Note that this includes all elements, even those that are 'hidden'.
     *
     * TODO: Can this replace getLinearDOM and goToNextItem?
     * 
     * @param  {DOMElement} element - parent element for which you want an array of all children
     * @return {array<DOMElement>}
     * @private
     */
    getLinearDOMElements: function( element ) {
      // gets all descendent children for the element 
      var children = element.getElementsByTagName( '*' );
      var linearDOM = [];
      for( var i = 0; i < children.length; i++ ) {
        if( children[i].nodeType === Node.ELEMENT_NODE ) {
          linearDOM[i] = ( children[ i ] );
        }
      }
      return linearDOM;
    },

    /**
     * Get a 'linear' representation of the DOM, collapsing the accessibility tree into an array that
     * can be traversed.
     *
     * @param {DOMElement} element
     * @return {array<DOMElement>} linearDOM
     * @private
     */
    getLinearDOM: function( element ) {

      this.clearVisited( element, DATA_VISITED_LINEAR );

      var linearDOM = [];

      var nextElement = this.goToNextItem( element, DATA_VISITED_LINEAR );
      while ( nextElement ) {
        linearDOM.push( nextElement );
        nextElement = this.goToNextItem( element, DATA_VISITED_LINEAR );
      }

      return linearDOM;
    },

    /**
     * Go to next element in the parallel DOM that has accessible text content.  If an element has
     * text content, it is returned.
     * 
     * @param {DOMElement} element
     * @return {string} visitedFlag - a flag for which 'data-*' attribute to set as we search through the tree
     * @private
     */
    goToNextItem: function( element, visitedFlag ) {
      if ( this.getAccessibleText( element ) ) {
        if ( !element.getAttribute( visitedFlag ) ) {
          element.setAttribute( visitedFlag, true );
          return element;
        }
      }
      for ( var i = 0; i < element.children.length; i++ ) {
        var nextElement = this.goToNextItem( element.children[ i ], visitedFlag );

        if ( nextElement ) {
          return nextElement;
        }
      }
    },

    /**
     * Update the list of live elements, and add a MutationObserver to any element that has an attribute
     * for aria-live.
     *
     * @private
     */
    updateLiveElementList: function() {

      var thisCursor = this;

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
      var linearDOM = this.getLinearDOMElements( accessibilityDOMElement );

      // search the DOM for elements with 'aria-live' attributes
      for( var i = 0; i < linearDOM.length; i++ ) {
        var domElement = linearDOM[ i ];
        if( domElement.getAttribute( 'aria-live' ) ) {

          // create an observer for this element
          var observer = new MutationObserver( function( mutations ) {
            mutations.forEach( function( mutation ) {
              var updatedText = mutation.addedNodes[0].data;
              thisCursor.ariaLiveQueue.add( function() {
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
    },

    /**
     * Clear the visited flag from all child elements under the parent element
     * 
     * @param  {DOMElememt} element
     * @param  {string} visitedFlag
     * @private
     */
    clearVisited: function( element, visitedFlag ) {
      element.removeAttribute( visitedFlag );
      for ( var i = 0; i < element.children.length; i++ ) {
        this.clearVisited( element.children[ i ], visitedFlag );
      }
    }

  } );

  /**
   * A timed queue that can be used to queue descriptions to the text output.  Usually used for
   * 'aria-live' updates.  If multiple updates are fired at once, they are added to the queue
   * in the order that they occur.
   * 
   * @param {number} defaultDelay - default delay for items in the queue
   */
  function TimedQueue( defaultDelay ) {

    // @private - track when we are running so we do not try to run the queue while it is already running
    this.running = false;

    // @private - the queue of items to be read, populated with objects like
    // { callback: {function}, delay: {number} };
    this.queue = [];

    // @private - current index in the queue
    this.index = 0;

    // @private - default delay of five seconds
    this.defaultDelay = defaultDelay || 5000;
  }

  scenery.register( 'TimedQueue', TimedQueue );

  inherit( Object, TimedQueue, {

    /**
     * Add a callback to this queue, with a delay
     * 
     * @param {function} callBack - callback fired by this addition
     * @param {number} [delay]- optional delay for this item
     * @public
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
     * @public
     */
    run: function( index ) {
      if ( !this.running ) {
        this.index = index || 0;
        this.next();
      }
    },

    /**
     * Remove all items from the queue
     * @public
     */
    clear: function() {
      this.queue = [];
    },

    /**
     * Get the next item in the queue, then delaying after firing callback
     * @public
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

  return VirtualCursor;

} );