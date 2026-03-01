import React from 'react'
import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar'
import Topbar from './Topbar'

export default function Layout() {
  return (
    <>
      <Sidebar />
      <Topbar />
      <div className="main-content">
        <div className="page-content">
          <Outlet />
        </div>
      </div>
    </>
  )
}
