// Copyright 2013-2015, University of Colorado Boulder

/**
 * DEPRECATED EXPERIMENTAL: USE AT YOUR OWN CAUTION
 *
 * A container that allows flexible layout generation based on layout methods that can be composed together.
 *
 * Experimental demo (for use in Scenery's playground):
 *   var n = new scenery.LayoutNode( scenery.LayoutNode.Vertical.and( scenery.LayoutNode.AlignLeft ) );
 *   scene.addChild( n );
 *   n.addChild( new scenery.Text( 'Some Text' ) );
 *   n.addChild( new scenery.Text( 'Text pushed to the right' ), { padLeft: 10 } );
 *   n.addChild( new scenery.Text( 'Some padding on top and bottom' ), { padTop: 10, padBottom: 20 } );
 *   n.addChild( new scenery.Text( 'Just Regular Text' ) );
 *   n.addChild( new scenery.Text( 'Right-aligned' ), { layoutMethod: scenery.LayoutNode.Vertical.and( scenery.LayoutNode.AlignRight ) } );
 *   n.addChild( new scenery.Text( 'Center-aligned' ), { layoutMethod: scenery.LayoutNode.Vertical.and( scenery.LayoutNode.AlignHorizontalCenter ) } );
 *   n.addChild( new scenery.Text( 'Pad from right' ), { layoutMethod: scenery.LayoutNode.Vertical.and( scenery.LayoutNode.AlignRight ), padRight: 10 } );
 *   n.children[2].text += ' and it updates!';
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var extend = require( 'PHET_CORE/extend' );
  var scenery = require( 'SCENERY/scenery' );
  var Node = require( 'SCENERY/nodes/Node' );
  // var Rectangle = require( 'SCENERY/nodes/Rectangle' );
  var Bounds2 = require( 'DOT/Bounds2' );

  // var debug = false;

  // @deprecated
  function LayoutNode( defaultMethod, options ) {
    var self = this;

    assert && assert( defaultMethod instanceof LayoutMethod, 'defaultMethod is required' );

    options = extend( {
      updateOnBounds: true,
      defaultMethod: defaultMethod
    }, options );

    this._activelyLayingOut = false;
    this._updateOnBounds = true;
    this._defaultMethod = null;
    this._elements = [];
    this._elementMap = {}; // maps node ID => element
    // this._invisibleBackground = new Rectangle( 0, 0, 0x1f, 0x1f, { visible: false } ); // takes up space that represents the bounds of all of the layout elements (with their padding)
    this._boundsListener = function() {
      if ( self._updateOnBounds ) {
        self.updateLayout();
      }
    };

    Node.call( this, options );

    // this.addChild( this._invisibleBackground );

    this.updateLayout();

    throw new Error( 'Deprecated, please do not use (replacement for overrideBounds has not been provided)' );
  }

  scenery.register( 'LayoutNode', LayoutNode );

  inherit( Node, LayoutNode, {
    get layoutProperties() { return new LayoutProperties( this._elements ); },

    get layoutBounds() {
      var result = Bounds2.NOTHING.copy();
      _.each( this._elements, function( element ) {
        result.includeBounds( element.layoutBounds );
      } );
      return result;
    },

    set defaultMethod( value ) {
      this._defaultMethod = value;
      this.updateLayout();
      return this;
    },
    get defaultMethod() { return this._defaultMethod; },

    set updateOnBounds( value ) {
      this._updateOnBounds = value;
      return this;
    },
    get updateOnBounds() { return this._updateOnBounds; },

    /*
     * Options can consist of:
     *   layoutMethod - layout method
     *   useVisibleBounds - (false) whether to use visible bounds instead of normal bounds - TODO how to auto-update when using these?
     *   padLeft / padRight / padTop / padBottom
     *   boundsMethod - custom overriding of entire boundsMethod
     */
    insertChild: function( index, node, options ) {
      var self = this;

      options = extend( {
        useVisibleBounds: false,
        padLeft: 0,
        padRight: 0,
        padTop: 0,
        padBottom: 0
      }, options );
      // var baseBoundsFunc = ( options.useVisibleBounds ? node.getVisibleBounds : node.getBounds ).bind( node );

      Node.prototype.insertChild.call( this, index, node );

      var methodGetter = options.layoutMethod ? function() { return options.layoutMethod; } : function() { return self._defaultMethod; };
      var element = new LayoutElement( node, methodGetter, options.boundsMethod ? options.boundsMethod : function( bounds ) {
        if ( options.useVisibleBounds ) {
          bounds = node.visibleBounds;
        }

        return new Bounds2( bounds.minX - options.padLeft, bounds.minY - options.padTop, bounds.maxX + options.padRight, bounds.maxY + options.padBottom );
      } );
      this.addElement( element );

      this.updateLayout();
    },

    addChild: function( node, options ) {
      this.insertChild( this._children.length, node, options );
    },

    // override
    removeChildWithIndex: function( node, indexOfChild ) {
      Node.prototype.removeChildWithIndex.call( this, node, indexOfChild );
      if ( this._elementMap[ node.id ] ) {
        delete this._elementMap[ node.id ];
      }

      this.updateLayout();
    },

    addElement: function( element ) {
      this._elements.push( element );
      element.node.addEventListener( 'bounds', this._boundsListener );
    },

    removeElement: function( element ) {
      this._elements.splice( this._elements.indexOf( element ), 1 ); // TODO: replace with some remove() instead of splice()
      element.node.removeEventListener( 'bounds', this._boundsListener );
    },

    updateLayout: function() {
      if ( this._activelyLayingOut ) {
        // don't start another layout while one is going on!
        return;
      }
      this._activelyLayingOut = true;
      var layoutProperties = this.layoutProperties;
      for ( var i = 0; i < this._elements.length; i++ ) {
        var element = this._elements[ i ];
        element.layoutMethod.layout( element, i, ( i > 0 ? this._elements[ i - 1 ] : null ), layoutProperties );
      }

      // use the invisible background to take up all of our layout areas
      // this._invisibleBackground.removeAllChildren();
      // var bounds = this.layoutBounds;
      // if ( !bounds.isEmpty() ) {
      //   this._invisibleBackground.addChild( new Rectangle( bounds, { visible: false } ) );
      // }

      // if ( debug ) {
      //   this._invisibleBackground.visible = true;
      //   this._invisibleBackground.fill = 'rgba(255,0,0,0.4)';

      //   _.each( this._elements, function( element ) {
      //     this._invisibleBackground.addChild( new Rectangle( element.node.bounds ), {
      //       fill: 'rgba(255,0,0,0.4)',
      //       stroke: 'blue'
      //     } );
      //   } );
      // }
      this._activelyLayingOut = false;
    }
  } );

  /*
   * LayoutMethod - function layout( element, index, previousElement, layoutProperties )
   */
  var LayoutMethod = LayoutNode.LayoutMethod = function LayoutMethod( layout ) {
    if ( layout ) {
      this.layout = layout;
    }
  };
  inherit( Object, LayoutMethod, {
    and: function( otherLayoutMethod ) {
      var self = this;

      return new LayoutMethod( function compositeLayout( element, index, previousElement, layoutProperties ) {
        self.layout( element, index, previousElement, layoutProperties );
        otherLayoutMethod.layout( element, index, previousElement, layoutProperties );
      } );
    }
  } );

  /*---------------------------------------------------------------------------*
   * Layout Methods
   *----------------------------------------------------------------------------*/

  LayoutNode.Vertical = new LayoutMethod( function verticalLayout( element, index, previousElement, layoutProperties ) {
    element.layoutTop = previousElement ? previousElement.layoutBounds.bottom : 0;
  } );

  LayoutNode.Horizontal = new LayoutMethod( function horizontalLayout( element, index, previousElement, layoutProperties ) {
    element.layoutLeft = previousElement ? previousElement.layoutBounds.right : 0;
  } );

  LayoutNode.AlignLeft = new LayoutMethod( function alignLeftLayout( element, index, previousElement, layoutProperties ) {
    element.layoutLeft = 0;
  } );

  LayoutNode.AlignHorizontalCenter = new LayoutMethod( function alignHorizontalCenterLayout( element, index, previousElement, layoutProperties ) {
    element.layoutLeft = ( layoutProperties.maxWidth - element.layoutBounds.width ) / 2;
  } );

  LayoutNode.AlignRight = new LayoutMethod( function alignRightLayout( element, index, previousElement, layoutProperties ) {
    element.layoutLeft = layoutProperties.maxWidth - element.layoutBounds.width;
  } );

  LayoutNode.AlignTop = new LayoutMethod( function alignTopLayout( element, index, previousElement, layoutProperties ) {
    element.layoutTop = 0;
  } );

  LayoutNode.AlignVerticalCenter = new LayoutMethod( function alignVerticalCenterLayout( element, index, previousElement, layoutProperties ) {
    element.layoutTop = ( layoutProperties.maxHeight - element.layoutBounds.height ) / 2;
  } );

  LayoutNode.AlignBottom = new LayoutMethod( function alignBottomLayout( element, index, previousElement, layoutProperties ) {
    element.layoutTop = layoutProperties.maxHeight - element.layoutBounds.height;
  } );

  /*---------------------------------------------------------------------------*
   * Internals
   *----------------------------------------------------------------------------*/

  var LayoutProperties = LayoutNode.LayoutProperties = function LayoutProperties( elements ) {
    var largestWidth = 0;
    var largestHeight = 0;

    _.each( elements, function( element ) {
      largestWidth = Math.max( largestWidth, element.layoutBounds.width );
      largestHeight = Math.max( largestHeight, element.layoutBounds.height );
    } );

    this.maxWidth = largestWidth;
    this.maxHeight = largestHeight;
  };

  var LayoutElement = LayoutNode.LayoutElement = function LayoutElement( node, layoutMethodGetter, boundsMethod ) {
    this.node = node;
    this.layoutMethodGetter = layoutMethodGetter;
    this.boundsMethod = boundsMethod;
  };
  inherit( Object, LayoutElement, {
    get bounds() { return this.node.bounds; },
    get layoutBounds() { return this.boundsMethod( this.bounds ); },
    get layoutMethod() { return this.layoutMethodGetter(); },

    get layoutTop() { throw new Error( 'JSHint wants this getter' ); },
    set layoutTop( y ) {
      var padding = this.bounds.top - this.layoutBounds.top;
      this.node.top = y + padding;
    },

    get layoutLeft() { throw new Error( 'JSHint wants this getter' ); },
    set layoutLeft( x ) {
      var padding = this.bounds.left - this.layoutBounds.left;
      this.node.left = x + padding;
    }
  } );

  LayoutNode.prototype._mutatorKeys = [ 'defaultMethod', 'updateOnBounds' ].concat( Node.prototype._mutatorKeys );

  return LayoutNode;
} );


