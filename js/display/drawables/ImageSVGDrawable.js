// Copyright 2016-2022, University of Colorado Boulder

/**
 * SVG drawable for Image nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../../phet-core/js/Poolable.js';
import { ImageStatefulDrawable, scenery, svgns, SVGSelfDrawable, xlinkns } from '../../imports.js';

// TODO: change this based on memory and performance characteristics of the platform
const keepSVGImageElements = true; // whether we should pool SVG elements for the SVG rendering states, or whether we should free them when possible for memory

class ImageSVGDrawable extends ImageStatefulDrawable( SVGSelfDrawable ) {
  /**
   * @public
   * @override
   *
   * @param {number} renderer
   * @param {Instance} instance
   */
  initialize( renderer, instance ) {
    super.initialize( renderer, instance, false, keepSVGImageElements ); // usesPaint: false

    sceneryLog && sceneryLog.ImageSVGDrawable && sceneryLog.ImageSVGDrawable( `${this.id} initialized for ${instance.toString()}` );

    // @protected {SVGImageElement} - Sole SVG element for this drawable, implementing API for SVGSelfDrawable
    this.svgElement = this.svgElement || document.createElementNS( svgns, 'image' );
    this.svgElement.setAttribute( 'x', '0' );
    this.svgElement.setAttribute( 'y', '0' );

    // @private {boolean} - Whether we have an opacity attribute specified on the DOM element.
    this.hasOpacity = false;

    // @private {boolean}
    this.usingMipmap = false;

    // @private {number} - will always be invalidated
    this.mipmapLevel = -1;

    // @private {function} - if mipmaps are enabled, this listener will be added to when our relative transform changes
    this._mipmapTransformListener = this._mipmapTransformListener || this.onMipmapTransform.bind( this );
    this._mipmapListener = this._mipmapListener || this.onMipmap.bind( this );

    this.node.mipmapEmitter.addListener( this._mipmapListener );
    this.updateMipmapStatus( instance.node._mipmap );
  }

  /**
   * Updates the SVG elements so that they will appear like the current node's representation.
   * @protected
   *
   * Implements the interface for SVGSelfDrawable (and is called from the SVGSelfDrawable's update).
   */
  updateSVGSelf() {
    const image = this.svgElement;

    if ( this.dirtyImage ) {
      sceneryLog && sceneryLog.ImageSVGDrawable && sceneryLog.ImageSVGDrawable( `${this.id} Updating dirty image` );
      if ( this.node._image ) {
        // like <image xlink:href='https://phet.colorado.edu/images/phet-logo-yellow.png' x='0' y='0' height='127px' width='242px'/>
        this.updateURL( image, true );
      }
      else {
        image.setAttribute( 'width', '0' );
        image.setAttribute( 'height', '0' );
        image.setAttributeNS( xlinkns, 'xlink:href', '//:0' ); // see http://stackoverflow.com/questions/5775469/whats-the-valid-way-to-include-an-image-with-no-src
      }
    }
    else if ( this.dirtyMipmap && this.node._image ) {
      sceneryLog && sceneryLog.ImageSVGDrawable && sceneryLog.ImageSVGDrawable( `${this.id} Updating dirty mipmap` );
      this.updateURL( image, false );
    }

    if ( this.dirtyImageOpacity ) {
      if ( this.node._imageOpacity === 1 ) {
        if ( this.hasOpacity ) {
          this.hasOpacity = false;
          image.removeAttribute( 'opacity' );
        }
      }
      else {
        this.hasOpacity = true;
        image.setAttribute( 'opacity', this.node._imageOpacity );
      }
    }
  }

  /**
   * @private
   *
   * @param {SVGImageElement} image
   * @param {boolean} forced
   */
  updateURL( image, forced ) {
    // determine our mipmap level, if any is used
    let level = -1; // signals a default of "we are not using mipmapping"
    if ( this.node._mipmap ) {
      level = this.node.getMipmapLevel( this.instance.relativeTransform.matrix );
      sceneryLog && sceneryLog.ImageSVGDrawable && sceneryLog.ImageSVGDrawable( `${this.id} Mipmap level: ${level}` );
    }

    // bail out if we would use the currently-used mipmap level (or none) and there was no image change
    if ( !forced && level === this.mipmapLevel ) {
      return;
    }

    // if we are switching to having no mipmap
    if ( this.mipmapLevel >= 0 && level === -1 ) {
      image.removeAttribute( 'transform' );
    }
    this.mipmapLevel = level;

    if ( this.node._mipmap && this.node.hasMipmaps() ) {
      sceneryLog && sceneryLog.ImageSVGDrawable && sceneryLog.ImageSVGDrawable( `${this.id} Setting image URL to mipmap level ${level}` );
      const url = this.node.getMipmapURL( level );
      const canvas = this.node.getMipmapCanvas( level );
      image.setAttribute( 'width', `${canvas.width}px` );
      image.setAttribute( 'height', `${canvas.height}px` );
      // Since SVG doesn't support parsing scientific notation (e.g. 7e5), we need to output fixed decimal-point strings.
      // Since this needs to be done quickly, and we don't particularly care about slight rounding differences (it's
      // being used for display purposes only, and is never shown to the user), we use the built-in JS toFixed instead of
      // Dot's version of toFixed. See https://github.com/phetsims/kite/issues/50
      image.setAttribute( 'transform', `scale(${Math.pow( 2, level ).toFixed( 20 )})` ); // eslint-disable-line bad-sim-text
      image.setAttributeNS( xlinkns, 'xlink:href', url );
    }
    else {
      sceneryLog && sceneryLog.ImageSVGDrawable && sceneryLog.ImageSVGDrawable( `${this.id} Setting image URL` );
      image.setAttribute( 'width', `${this.node.getImageWidth()}px` );
      image.setAttribute( 'height', `${this.node.getImageHeight()}px` );
      image.setAttributeNS( xlinkns, 'xlink:href', this.node.getImageURL() );
    }
  }

  /**
   * @private
   *
   * @param {boolean} usingMipmap
   */
  updateMipmapStatus( usingMipmap ) {
    if ( this.usingMipmap !== usingMipmap ) {
      this.usingMipmap = usingMipmap;

      if ( usingMipmap ) {
        sceneryLog && sceneryLog.ImageSVGDrawable && sceneryLog.ImageSVGDrawable( `${this.id} Adding mipmap compute/listener needs` );
        this.instance.relativeTransform.addListener( this._mipmapTransformListener ); // when our relative tranform changes, notify us in the pre-repaint phase
        this.instance.relativeTransform.addPrecompute(); // trigger precomputation of the relative transform, since we will always need it when it is updated
      }
      else {
        sceneryLog && sceneryLog.ImageSVGDrawable && sceneryLog.ImageSVGDrawable( `${this.id} Removing mipmap compute/listener needs` );
        this.instance.relativeTransform.removeListener( this._mipmapTransformListener );
        this.instance.relativeTransform.removePrecompute();
      }

      // sanity check
      this.markDirtyMipmap();
    }
  }

  /**
   * @private
   */
  onMipmap() {
    // sanity check
    this.markDirtyMipmap();

    // update our mipmap usage status
    this.updateMipmapStatus( this.node._mipmap );
  }

  /**
   * @private
   */
  onMipmapTransform() {
    sceneryLog && sceneryLog.ImageSVGDrawable && sceneryLog.ImageSVGDrawable( `${this.id} Transform dirties mipmap` );

    this.markDirtyMipmap();
  }

  /**
   * Disposes the drawable.
   * @public
   * @override
   */
  dispose() {
    sceneryLog && sceneryLog.ImageSVGDrawable && sceneryLog.ImageSVGDrawable( `${this.id} disposing` );

    // clean up mipmap listeners and compute needs
    this.updateMipmapStatus( false );
    this.node.mipmapEmitter.removeListener( this._mipmapListener );

    super.dispose();
  }
}

scenery.register( 'ImageSVGDrawable', ImageSVGDrawable );

Poolable.mixInto( ImageSVGDrawable );

export default ImageSVGDrawable;