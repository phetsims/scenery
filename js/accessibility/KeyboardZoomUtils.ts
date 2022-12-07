// Copyright 2019-2022, University of Colorado Boulder

/**
 * Utilities specific to the keyboard for handling zoom/pan control.
 *
 * @author Jesse Greenberg
 */

import { KeyboardUtils, scenery } from '../imports.js';

const KeyboardZoomUtils = {

  /**
   * Returns true if the platform is most likely a Mac device. Pan/Zoom will use different modifier keys in this case.
   *
   * TODO: Move to platform if generally useful?
   */
  isPlatformMac: (): boolean => {
    return _.includes( window.navigator.platform, 'Mac' );
  },

  /**
   * Get the 'meta' key for the platform that would indicate user wants to zoom. This is 'metaKey' on Mac and 'ctrl'
   * on Windows.
   *
   */
  getPlatformZoomMetaKey: (): string => {
    return KeyboardZoomUtils.isPlatformMac() ? 'metaKey' : 'ctrlKey';
  },

  /**
   * Returns true of the keyboard input indicates that a zoom command was initiated. Different keys are checked
   * on mac devices (which go through the Cmd key) and windows devices (which use the ctrl modifier).
   *
   * @param event
   * @param zoomIn - do you want to check for zoom in or zoom out?
   */
  isZoomCommand: ( event: Event, zoomIn: boolean ): boolean => {
    const zoomKey = zoomIn ? KeyboardUtils.KEY_EQUALS : KeyboardUtils.KEY_MINUS;
    const metaKey = KeyboardZoomUtils.getPlatformZoomMetaKey();

    // @ts-expect-error
    return event[ metaKey ] && KeyboardUtils.isKeyEvent( event, zoomKey );
  },

  /**
   * Returns true if the keyboard command indicates a "zoom reset". This is ctrl + 0 on Win and cmd + 0 on mac.
   */
  isZoomResetCommand: ( event: Event ): boolean => {
    const metaKey = KeyboardZoomUtils.getPlatformZoomMetaKey();

    // @ts-expect-error
    return event[ metaKey ] && KeyboardUtils.isKeyEvent( event, KeyboardUtils.KEY_0 );
  }
};

scenery.register( 'KeyboardZoomUtils', KeyboardZoomUtils );
export default KeyboardZoomUtils;