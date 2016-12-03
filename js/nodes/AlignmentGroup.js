// Copyright 2016-2016, University of Colorado Boulder

/**
 * A group of containers that follow the constraints:
 * 1. Every container will have the same bounds, with an upper-left of (0,0)
 * 2. The container sizes will be the smallest possible to fit every container's content (with respective padding).
 * 3. Each container is responsible for positioning its content in its bounds (with customizable alignment and padding).
 *
 * Containers can be dynamically created and disposed, and only active containers will be considered for the bounds.
 *
 * NOTE: Container resizes may not happen immediately, and may be delayed until bounds of a container's child occurs.
 *       layout updates can be forced with group.updateLayout(). If the container's content that changed is connected
 *       to a Scenery display, its bounds will update when Display.updateDisplay() will called, so this will guarantee
 *       that the layout will be applied before it is displayed.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

define( function( require ) {
  'use strict';

  var inherit = require( 'PHET_CORE/inherit' );
  var arrayRemove = require( 'PHET_CORE/arrayRemove' );
  var Bounds2 = require( 'DOT/Bounds2' );
  var scenery = require( 'SCENERY/scenery' );
  var Node = require( 'SCENERY/nodes/Node' );

  /**
   * Creates an alignment group that can be composed of multiple containers.
   * @constructor
   * @public
   *
   * Use createContainer() to create containers. You can dispose() individual containers, or call dispose() on this
   * group to dispose all of them.
   */
  function AlignmentGroup() {
    // @private {Array.<AlignmentContainer>}
    this._containers = [];

    // @private {boolean} - Gets locked when certain layout is performed.
    this._resizeLock = false;
  }

  scenery.register( 'AlignmentGroup', AlignmentGroup );

  inherit( Object, AlignmentGroup, {
    /**
     * Creates a container with the given content and options.
     * @public
     *
     * @param {Node} content - Note that the content may be repositioned into place.
     * @param {Object} [options] - See AlignmentContainer's constructor below for specific alignment options.
     *                             Will also be passed to the node's constructor.
     */
    createContainer: function( content, options ) {
      assert && assert( content instanceof Node );

      // Set a resize lock around creating the alignment container, so that we don't waste time on every mutation.
      // We'll want to restore its previous value afterwards.
      var lastLock = this._resizeLock;
      this._resizeLock = true;
      var container = new AlignmentContainer( content, this, options );
      this._resizeLock = lastLock;

      this._containers.push( container );

      // Trigger an update when a container is added
      this.updateLayout();

      return container;
    },

    /**
     * Dispose all of the containers.
     * @public
     */
    dispose: function() {
      for ( var i = this._containers.length - 1; i >= 0; i-- ) {
        this._containers[ i ].dispose();
      }
    },

    /**
     * Updates the localBounds and alignment for each container.
     * @public
     *
     * NOTE: Calling this will usually not be necessary outside of Scenery, but this WILL trigger bounds revalidation
     *       for every container, which can force the layout code to run.
     */
    updateLayout: function() {
      if ( this._resizeLock ) { return; }
      this._resizeLock = true;

      // Compute the maximum dimension of our containers' content
      var maxWidth = 0;
      var maxHeight = 0;
      for ( var i = 0; i < this._containers.length; i++ ) {
        var container = this._containers[ i ];

        var bounds = container.getContentBounds();

        // Ignore bad bounds
        if ( bounds.isEmpty() || !bounds.isFinite() ) {
          continue;
        }

        maxWidth = Math.max( maxWidth, bounds.width );
        maxHeight = Math.max( maxHeight, bounds.height );
      }

      if ( maxWidth > 0 && maxHeight > 0 ) {
        // Apply that maximum dimension for each container
        for ( i = 0; i < this._containers.length; i++ ) {
          this._containers[ i ].updateLayout( maxWidth, maxHeight );
        }
      }

      this._resizeLock = false;
    },

    /**
     * Lets the group know that the container has had its content resized.
     * @private
     *
     * @param {AlignmentContainer}
     */
    onContainerContentResized: function( container ) {
      // TODO: in the future, we could only update this specific container if the others don't need updating.
      this.updateLayout();
    },

    /**
     * Handles disposal of the container
     * @private
     */
    disposeContainer: function( container ) {
      arrayRemove( this._containers, container );

      // Trigger an update when a container is removed
      this.updateLayout();
    }
  } );

  var ALIGNMENT_CONTAINER_OPTION_KEYS = [
    'xAlign', // {string} - 'left', 'center', or 'right', for horizontal positioning
    'yAlign', // {string} - 'top', 'center', or 'bottom', for vertical positioning
    'xMargin', // {number} - Non-negative margin that should exist on the left and right of the content
    'yMargin' // {number} - Non-negative margin that should exist on the top and bottom of the content.
  ];

  /**
   * An individual container for an alignment group. Will maintain its size to match that of the group by overriding
   * its localBounds, and will position its content inside its localBounds by respecting its alignment and margins.
   * @constructor
   * @private
   *
   * @param {Node} content - Node that was passed to createContainer(). We will change its position to keep it aligned.
   * @param {AlignmentGroup} alignmentGroup
   * @param {Object} [options] - See the _.extend below in AlignmentContainer for documentation. Also passed to Node.
   */
  function AlignmentContainer( content, alignmentGroup, options ) {
    // @private {string}
    this._xAlign = 'center';
    this._yAlign = 'center';

    // @private {number}
    this._xMargin = 0;
    this._yMargin = 0;

    // @private {Node}
    this._content = content;

    // @private {AlignmentGroup}
    this._alignmentGroup = alignmentGroup;

    // @private {function} - Callback for when bounds change
    this._contentBoundsListener = alignmentGroup.onContainerContentResized.bind( alignmentGroup, this );

    // dispose() will remove it.
    this._content.on( 'bounds', this._contentBoundsListener );

    Node.call( this, _.extend( {}, options, {
      children: [ this._content ]
    } ) );
  }

  inherit( Node, AlignmentContainer, {
    /**
     * {Array.<string>} - String keys for all of the allowed options that will be set by node.mutate( options ), in the
     * order they will be evaluated in.
     * @protected
     *
     * NOTE: See Node's _mutatorKeys documentation for more information on how this operates, and potential special
     *       cases that may apply.
     */
    _mutatorKeys: ALIGNMENT_CONTAINER_OPTION_KEYS.concat( Node.prototype._mutatorKeys ),

    /**
     * Sets the horizontal alignment of this container.
     * @public
     *
     * Available values are 'left', 'center', or 'right'.
     *
     * @param {string} xAlign
     * @returns {AlignmentContainer} - For chaining
     */
    setXAlign: function( xAlign ) {
      assert && assert( xAlign === 'left' || xAlign === 'center' || xAlign === 'right',
        'xAlign should be one of: \'left\', \'center\', or \'right\'' );

      if ( this._xAlign !== xAlign ) {
        this._xAlign = xAlign;

        // Trigger re-layout
        this._alignmentGroup.onContainerContentResized( this );
      }

      return this;
    },
    set xAlign( value ) { this.setXAlign( value ); },

    /**
     * Returns the current horizontal alignment of this container.
     * @public
     *
     * @returns {string} - See setXAlign for values.
     */
    getXAlign: function() {
      return this._xAlign;
    },
    get xAlign() { return this.getXAlign(); },

    /**
     * Sets the vertical alignment of this container.
     * @public
     *
     * Available values are 'top', 'center', or 'bottom'.
     *
     * @param {string} yAlign
     * @returns {AlignmentContainer} - For chaining
     */
    setYAlign: function( yAlign ) {
      assert && assert( yAlign === 'top' || yAlign === 'center' || yAlign === 'bottom',
        'yAlign should be one of: \'top\', \'center\', or \'bottom\'' );

      if ( this._yAlign !== yAlign ) {
        this._yAlign = yAlign;

        // Trigger re-layout
        this._alignmentGroup.onContainerContentResized( this );
      }

      return this;
    },
    set yAlign( value ) { this.setYAlign( value ); },

    /**
     * Returns the current vertical alignment of this container.
     * @public
     *
     * @returns {string} - See setYAlign for values.
     */
    getYAlign: function() {
      return this._yAlign;
    },
    get yAlign() { return this.getYAlign(); },

    /**
     * Sets the horizontal margin of this container.
     * @public
     *
     * This margin is the minimum amount of horizontal space that will exist between the content and the left and
     * right sides of this container.
     *
     * @param {number} xMargin
     * @returns {AlignmentContainer} - For chaining
     */
    setXMargin: function( xMargin ) {
      assert && assert( typeof xMargin === 'number' && isFinite( xMargin ) && xMargin >= 0,
        'xMargin should be a finite non-negative number' );

      if ( this._xMargin !== xMargin ) {
        this._xMargin = xMargin;

        // Trigger re-layout
        this._alignmentGroup.onContainerContentResized( this );
      }

      return this;
    },
    set xMargin( value ) { this.setXMargin( value ); },

    /**
     * Returns the current horizontal margin of this container.
     * @public
     *
     * @returns {number} - See setXMargin for values.
     */
    getXMargin: function() {
      return this._xMargin;
    },
    get xMargin() { return this.getXMargin(); },

    /**
     * Sets the vertical margin of this container.
     * @public
     *
     * This margin is the minimum amount of vertical space that will exist between the content and the top and
     * bottom sides of this container.
     *
     * @param {number} yMargin
     * @returns {AlignmentContainer} - For chaining
     */
    setYMargin: function( yMargin ) {
      assert && assert( typeof yMargin === 'number' && isFinite( yMargin ) && yMargin >= 0,
        'yMargin should be a finite non-negative number' );

      if ( this._yMargin !== yMargin ) {
        this._yMargin = yMargin;

        // Trigger re-layout
        this._alignmentGroup.onContainerContentResized( this );
      }

      return this;
    },
    set yMargin( value ) { this.setYMargin( value ); },

    /**
     * Returns the current vertical margin of this container.
     * @public
     *
     * @returns {number} - See setYMargin for values.
     */
    getYMargin: function() {
      return this._yMargin;
    },
    get yMargin() { return this.getYMargin(); },

    /**
     * Returns the bounding box of this container's content. This will include any margins.
     * @private
     *
     * @returns {Bounds2}
     */
    getContentBounds: function() {
      return this._content.bounds.dilatedXY( this._xMargin, this._yMargin );
    },

    /**
     * Updates the layout of this alignment container.
     * @private
     *
     * @param {number} width
     * @param {number} height
     */
    updateLayout: function( width, height ) {
      // Avoid layout if width/height haven't been properly updated yet
      if ( width === 0 && height === 0 ) {
        return;
      }

      this.localBounds = new Bounds2( 0, 0, width, height );

      if ( this._xAlign === 'center' ) {
        this._content.centerX = this.localBounds.centerX;
      }
      else if ( this._xAlign === 'left' ) {
        this._content.left = this.localBounds.left + this._xMargin;
      }
      else if ( this._xAlign === 'right' ) {
        this._content.right = this.localBounds.right - this._xMargin;
      }
      else {
        assert && assert( 'Bad xAlign: ' + this._xAlign );
      }

      if ( this._yAlign === 'center' ) {
        this._content.centerY = this.localBounds.centerY;
      }
      else if ( this._yAlign === 'top' ) {
        this._content.top = this.localBounds.top + this._yMargin;
      }
      else if ( this._yAlign === 'bottom' ) {
        this._content.bottom = this.localBounds.bottom - this._yMargin;
      }
      else {
        assert && assert( 'Bad yAlign: ' + this._yAlign );
      }

      assert && assert( this.localBounds.containsBounds( this._content.bounds ),
        'All of our contents should be contained in our localBounds' );
    },

    /**
     * Disposes this container, so that the alignment group won't update this container, and won't use its bounds for
     * laying out the other containers.
     * @public
     */
    dispose: function() {
      this._content.off( 'bounds', this._contentBoundsListener );

      this._alignmentGroup.disposeContainer( this );
    }
  } );

  return AlignmentGroup;
} );
