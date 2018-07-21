// Copyright 2017, University of Colorado Boulder

/**
 * A method that is meant to be composed with Node. This method is responsible for updating DOM Elements the
 * AccessiblePeer when A11y settings on the Node change. This method should only be called when it is necessary to
 * recreate DOM elements. If small changes, or content changes are needed than it is preferable to just updating
 * those pieces manually (better performance).
 *
 * This file is responsible for creating the AccessiblePeer for each AccessibleInstance of Node.
 *
 * NOTE: This function assumes that "this" is of type {Node}, such that this function is added to a Node's prototype.
 *
 * @author Michael Kauzmann (PhET Interactive Simulations)
 * @author Jesse Greenberg (PhET Interactive Simulations)
 */

define( function( require ) {
  'use strict';

  // modules
  var AccessibilityUtil = require( 'SCENERY/accessibility/AccessibilityUtil' );
  var AccessiblePeer = require( 'SCENERY/accessibility/AccessiblePeer' );
  var scenery = require( 'SCENERY/scenery' );

  // constants
  var A_TAG = AccessibilityUtil.TAGS.A;
  var INPUT_TAG = AccessibilityUtil.TAGS.INPUT;

  // these elements require a minimum width to be visible in Safari
  var ELEMENTS_REQUIRE_WIDTH = [ INPUT_TAG, A_TAG ];

  /**
   * Invalidate our current accessible content, triggering recomputation
   * of anything that depended on the old accessible content. This can be
   * combined with a client (subType) implementation of invalidateAccessibleContent as well.
   *
   * Set's the `accessibleContent` of "this" (hopefully of type {Node}).
   *
   * @protected
   */
  var invalidateAccessibleContent = function() {
    var self = this;

    // iteration variable used through this function
    var i = 0;

    // for each accessible peer, clear the container parent if it exists since we will be reinserting labels and
    // the dom element in createPeer
    for ( var j = 0; j < this._accessibleInstances.length; j++ ) {
      var peer = this._accessibleInstances[ j ].peer;
      if ( peer ) {
        var containerElement = peer.containerParent;
        while ( containerElement && containerElement.hasChildNodes() ) {
          containerElement.removeChild( containerElement.lastChild );
        }
      }
    }

    // if any parents are flagged as removed from the accessibility tree, set content to null
    var contentDisplayed = this._accessibleContentDisplayed;
    for ( i = 0; i < this._parents.length; i++ ) {
      if ( !this._parents[ i ].accessibleContentDisplayed ) {
        contentDisplayed = false;
      }
    }

    var accessibleContent = null;
    if ( contentDisplayed && this._tagName ) {
      accessibleContent = {
        createPeer: function( accessibleInstance ) {

          // higher level api first, because it will effect the lower level setters.
          if ( self.accessibleName ) {
            self.setAccessibleNameImplementation( self.accessibleName ); // set it again to support any option order
          }

          if ( self.helpText ) {
            self.setHelpTextImplementation( self.helpText ); // set it again to support any option order
          }

          var uniqueId = accessibleInstance.trail.getUniqueId();

          // create the base DOM element representing this accessible instance
          var primarySibling = createElement( self._tagName, self.focusable, {
            namespace: self._accessibleNamespace
          } );
          primarySibling.id = uniqueId;

          // create the container parent for the dom siblings
          var containerElement = null;
          if ( self._containerTagName ) {
            containerElement = createElement( self._containerTagName, false );
            containerElement.id = 'container-' + uniqueId;

            // provide the aria-role if it is specified
            if ( self._containerAriaRole ) {
              containerElement.setAttribute( 'role', self._containerAriaRole );
            }
          }

          // create the label DOM element representing this instance
          var labelSibling = null;
          if ( self._labelTagName ) {
            labelSibling = createElement( self._labelTagName, false );
            labelSibling.id = 'label-' + uniqueId;
          }

          // create the description DOM element representing this instance
          var descriptionSibling = null;
          if ( self._descriptionTagName ) {
            descriptionSibling = createElement( self._descriptionTagName, false );
            descriptionSibling.id = 'description-' + uniqueId;
          }

          var accessiblePeer = AccessiblePeer.createFromPool( accessibleInstance, primarySibling, {
            containerParent: containerElement,
            labelSibling: labelSibling,
            descriptionSibling: descriptionSibling
          } );
          accessibleInstance.peer = accessiblePeer;

          // set the accessible label now that the element has been recreated again, but not if the tagName
          // has been cleared out
          if ( self._labelContent && self._labelTagName !== null ) {
            accessiblePeer.setLabelSiblingContent( self._labelContent );
          }

          // restore the innerContent
          if ( self._innerContent && self._tagName !== null ) {
            accessiblePeer.setPrimarySiblingContent( self._innerContent );
          }

          // set the accessible description, but not if the tagName has been cleared out.
          if ( self._descriptionContent && self._descriptionTagName !== null ) {
            accessiblePeer.setDescriptionSiblingContent( self._descriptionContent );
          }

          // set if using aria-label
          self._ariaLabel && accessiblePeer.setAttributeToElement( 'aria-label', self._ariaLabel );

          // restore visibility
          self.setAccessibleVisible( self._accessibleVisible );

          // restore checked
          accessiblePeer.setAttributeToElement( 'checked', self._accessibleChecked );

          // restore input value
          self._inputValue && accessiblePeer.setAttributeToElement( 'value', self._inputValue );


          // set the accessible attributes, restoring from a defensive copy
          var defensiveAttributes = self.accessibleAttributes;
          for ( i = 0; i < defensiveAttributes.length; i++ ) {
            var attribute = defensiveAttributes[ i ].attribute;
            var value = defensiveAttributes[ i ].value;
            var namespace = defensiveAttributes[ i ].namespace;
            accessiblePeer.setAttributeToElement( attribute, value, {
              namespace: namespace
            } );
          }

          // if element is an input element, set input type
          if ( self._tagName.toUpperCase() === INPUT_TAG && self._inputType ) {
            accessiblePeer.setAttributeToElement( 'type', self._inputType );
          }

          // recompute and assign the association attributes that link two elements (like aria-labelledby)
          self.updateLabelledbyDescribedbyAssociations();

          // add all listeners to the dom element
          for ( i = 0; i < self._accessibleInputListeners.length; i++ ) {
            accessiblePeer.addDOMEventListeners( self._accessibleInputListeners[ i ] );
          }

          // insert the label and description elements in the correct location if they exist
          labelSibling && accessiblePeer.arrangeContentElement( labelSibling, self._appendLabel );
          descriptionSibling && accessiblePeer.arrangeContentElement( descriptionSibling, self._appendDescription );

          // Default the focus highlight in this special case to be invisible until selected.
          if ( self._focusHighlightLayerable ) {
            self._focusHighlight.visible = false;
          }

          return accessiblePeer;
        }
      };
    }

    this.accessibleContent = accessibleContent;
  };


  /**
   * Create an HTML element.  Unless this is a form element or explicitly marked as focusable, add a negative
   * tab index. IE gives all elements a tabIndex of 0 and handles tab navigation internally, so this marks
   * which elements should not be in the focus order.
   *
   * @param  {string} tagName
   * @param {boolean} focusable - should the element be explicitly added to the focus order?
   * @param {Object} [options]
   * @returns {HTMLElement}
   */
  function createElement( tagName, focusable, options ) {
    options = _.extend( {
      namespace: null // {string|null} - If non-null, the element will be created with the specific namespace
    }, options );

    var domElement = options.namespace
                     ? document.createElementNS( options.namespace, tagName )
                     : document.createElement( tagName );
    var upperCaseTagName = tagName.toUpperCase();

    domElement.tabIndex = focusable ? 0 : -1;

    // Safari requires that certain input elements have dimension, otherwise it will not be keyboard accessible
    if ( _.includes( ELEMENTS_REQUIRE_WIDTH, upperCaseTagName ) ) {
      domElement.style.width = '1px';
      domElement.style.height = '1px';
    }

    return domElement;
  }

  scenery.register( 'invalidateAccessibleContent', invalidateAccessibleContent );


  return invalidateAccessibleContent;
} );