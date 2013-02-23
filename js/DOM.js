// Copyright 2002-2012, University of Colorado

/**
 * DOM nodes. Currently lightweight handling
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

define( function( require ) {
  "use strict";
  
  var assert = require( 'ASSERT/assert' )( 'scenery' );
  
  var Bounds2 = require( 'DOT/Bounds2' );
  
  var Node = require( 'SCENERY/Node' );
  var LayerType = require( 'SCENERY/LayerType' );
  var objectCreate = require( 'SCENERY/Util' ).objectCreate;
  
  var DOM = function( element, options ) {
    options = options || {};
    
    // unwrap from jQuery if that is passed in, for consistency
    if ( element && element.jquery ) {
      element = element[0];
    }
    
    this._element = element;
    this._$element = $( element );
    this._$element.css( 'position', 'absolute' );
    this._$element.css( 'left', 0 );
    this._$element.css( 'top', 0 );
    
    // so that the mutator will call setElement()
    options.element = element;
    
    // create a temporary container attached to the DOM tree (not a fragment) so that we can properly set initial bounds
    var temporaryContainer = document.createElement( 'div' );
    $( temporaryContainer ).css( {
      display: 'hidden',
      padding: '0 !important',
      margin: '0 !important',
      position: 'absolute',
      left: 0,
      top: 0
    } );
    temporaryContainer.appendChild( element );
    document.body.appendChild( temporaryContainer );
    
    // will set the element after initializing
    Node.call( this, options );
    
    // now don't memory-leak our container
    document.body.removeChild( temporaryContainer );
    if ( element.parentNode === temporaryContainer ) {
      temporaryContainer.removeChild( element );
    }
  };
  
  DOM.prototype = objectCreate( Node.prototype );
  DOM.prototype.constructor = DOM;
  
  DOM.prototype.invalidatePaint = function( bounds ) {
    Node.prototype.invalidatePaint.call( this, bounds );
  };
  
  DOM.prototype.invalidateDOM = function() {
    // TODO: do we need to reset the CSS transform to get the proper bounds?
    
    // TODO: reset with the proper bounds here
    this.invalidateSelf( new Bounds2( 0, 0, this._$element.width(), this._$element.height() ) );
  };
  
  DOM.prototype.addToDOMLayer = function( domLayer ) {
    // TODO: find better way to handle non-jquery and jquery-wrapped getters for the container. direct access for now ()
    domLayer.$div.append( this._element );
    
    // recompute the bounds
    this.invalidateDOM();
  };
  
  DOM.prototype.updateCSSTransform = function( transform ) {
    this._$element.css( transform.getMatrix().cssTransformStyles() );
  };
  
  DOM.prototype.hasSelf = function() {
    return true;
  };
  
  DOM.prototype.setElement = function( element ) {
    this._element = element;
    
    // TODO: bounds issue, since this will probably set to empty bounds and thus a repaint may not draw over it
    this.invalidateDOM();
    
    return this; // allow chaining
  };
  
  DOM.prototype.getElement = function() {
    return this._element;
  };
  
  DOM.prototype._mutatorKeys = [ 'element' ].concat( Node.prototype._mutatorKeys );
  
  DOM.prototype._supportedLayerTypes = [ LayerType.DOM ];
  
  Object.defineProperty( DOM.prototype, 'element', { set: DOM.prototype.setElement, get: DOM.prototype.getElement } );
  
  return DOM;
} );


